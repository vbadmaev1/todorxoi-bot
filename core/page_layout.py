# -*- coding: utf-8 -*-
"""
page_layout.py — разметка страницы с вертикальным письмом: скан или фото ->
столбцы, которые по одному читает модель распознавания (core/ocr.py).

Модель обучена на отдельных столбцах. Первая версия нарезки (split_columns
из архива модели) искала между столбцами совершенно пустые полосы шириной
от 20 px. На настоящей странице их нет: хвосты букв заходят под соседний
столбец, между столбцами пыль скана, колонтитул и номер страницы идут
поперёк — и вся страница уходила в модель одним «столбцом». Здесь иначе:

1. перевод в серое с учётом цвета (to_gray: текст тёмный, фон светлый),
   бинаризация (для фото — с выравниванием фона); пыль — компоненты меньше
   трети типичной точки на этой же картинке; сплошные пятна (стол вокруг
   страницы, тень, чёрная полоса) — тоже прочь;
2. наклон: угол, при котором вертикальная проекция чернил самая «резкая»
   (±5°);
3. столбцы — по стержням букв: у тодо бичиг в каждом столбце вертикальная
   ось, в проекции это высокий пик; граница между соседними столбцами —
   самая низкая точка проекции между пиками. Порогов в пикселях нет,
   поэтому разрешение не важно;
4. связные компоненты (буквы, слова) раздаются столбцам целиком — где у
   компоненты больше чернил, туда и идёт, поэтому хвост, заехавший за
   границу, не отрезается и не попадает к соседу. Слипшиеся — режутся;
5. у страницы (от 4 столбцов) выкидываются колонтитул, заголовок, номер
   страницы; у любой картинки — клочки, далеко оторванные от текста;
6. столбец собирается заново: серые пиксели исходника под маской только
   его компонент (и бледных волосных линий, которые от них отходят),
   остальное белое, поля pad × толщина — как в рамках, на которых училась
   модель.

Проверено на синтетическом тесте (3422 картинки, 1–3 столбца: CER 0.109%
против 0.110% у split_columns) и на «Калмыцких сказках» Позднеева (1889):
там split_columns находил один столбец на страницу, split_page — все 15.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks

_EIGHT = np.ones((3, 3), bool)


# ---------------------------------------------------------------- бинаризация

def _otsu(v):
    hist = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return int(np.argmax(between))


def _separability(v):
    """Насколько хорошо значения делятся на два класса (текст и фон): доля
    межклассовой дисперсии при пороге Оцу, от 0 до 1."""
    v = np.clip(v, 0, 255).astype(np.uint8)
    t = _otsu(v)
    lo, hi = v[v <= t], v[v > t]
    if not len(lo) or not len(hi) or v.var() == 0:
        return 0.0
    w = len(lo) / v.size
    return float(w * (1 - w) * (hi.mean() - lo.mean()) ** 2 / v.var())


def _stretch(x, sample):
    lo, hi = np.percentile(sample, [0.5, 99.5])
    if hi - lo < 1:
        return np.full(x.shape, 255, np.uint8)
    return np.clip((x - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8)


def to_gray(img):
    """Любая картинка -> серое uint8, где текст тёмный, а фон светлый.

    Раньше здесь было img.convert("L") и «фон тёмный, если края темнее 128».
    На цветном это ломалось двумя способами:
      • полярность — жёлтый текст на оранжевом в сером даёт 227 на 155: текст
        светлее фона, но и фон «светлый» (больше 128), поэтому текст шёл за
        фон, а за буквы принимались ореолы вокруг них. Так же чёрное на синем
        (фон 29 — «тёмный», переворачивалось зря);
      • контраст — у красного на зелёном яркость почти одинаковая, в сером
        букв не видно вовсе.
    Поэтому:
      1. прозрачность: текст на прозрачном фоне — это сам альфа-канал;
         прозрачные поля вокруг картинки заливаются её же цветом;
      2. из двух проекций цвета — яркость и главная ось разброса цветов
         (PCA по RGB) — берётся та, что лучше делит пиксели на два класса;
      3. текст — класс из тонких штрихов (см. _text_is_light); его делаем
         тёмным, фон светлым.
    Из 100 сочетаний цветов текста и фона из палитры бота раньше читались 69.
    """
    if img.mode in ("1", "L", "I", "I;16", "F"):
        g = np.asarray(img.convert("L"), dtype=np.float32)
        candidates = [g]
        k = max(1, round((g.size / 250_000) ** 0.5))
        samples = [g[::k, ::k].ravel()]
    else:
        has_alpha = "A" in img.mode or "transparency" in img.info
        rgba = np.asarray(img.convert("RGBA" if has_alpha else "RGB"), dtype=np.float32)
        rgb = rgba[..., :3]
        if has_alpha:
            alpha = rgba[..., 3:] / 255.0
            opaque = alpha[..., 0] > 0.5
            if 0 < opaque.mean() < 0.5:
                # непрозрачного мало — это и есть надпись на прозрачном фоне
                return (255 - alpha[..., 0] * 255).astype(np.uint8)
            if opaque.mean() < 1:
                fill = np.median(rgb[opaque], axis=0) if opaque.any() else np.full(3, 255.0)
                rgb = rgb * alpha + fill * (1 - alpha)
        k = max(1, round((rgb.shape[0] * rgb.shape[1] / 250_000) ** 0.5))
        pix = rgb[::k, ::k].reshape(-1, 3)
        weights = np.array([0.299, 0.587, 0.114], np.float32)
        candidates, samples = [rgb @ weights], [pix @ weights]
        cov = np.cov(pix.T)
        if np.trace(cov) > 1:
            axis = np.linalg.eigh(cov)[1][:, -1].astype(np.float32)   # направление наибольшего разброса
            candidates.append(rgb @ axis)
            samples.append(pix @ axis)
    stretched = [_stretch(s, s) for s in samples]
    scores = [_separability(s) for s in stretched]
    # яркость — если цветовая ось не заметно лучше: у обычных фото и сканов так надёжнее
    best = 1 if len(scores) > 1 and scores[1] > scores[0] + 0.05 else 0
    g = _stretch(candidates[best], samples[best])
    return 255 - g if _text_is_light(g) else g


def _median_run(b):
    """Медиана длин горизонтальных отрезков True в двумерном массиве."""
    d = np.diff(np.pad(b, ((0, 0), (1, 1))).astype(np.int8), axis=1).ravel()
    runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return float(np.median(runs)) if len(runs) else 0.0


def _text_is_light(g):
    """Текст — тот из двух классов, что состоит из тонких штрихов: его
    горизонтальные отрезки — поперечники штрихов, а у фона — промежутки между
    буквами и поля, они длиннее. Ни «чего больше», ни «какого цвета край» так
    не работают: на фото страницы тёмный стол вокруг занимает полкадра и весь
    край. Если отрезки почти равны — фон тот, чего больше."""
    k = max(1, round((g.size / 1_000_000) ** 0.5))
    small = g[::k, ::k]
    light = small > _otsu(small)
    dark_run, light_run = _median_run(~light), _median_run(light)
    if dark_run and light_run and not 0.8 < dark_run / light_run < 1.25:
        return light_run < dark_run
    return np.count_nonzero(light) < 0.5 * light.size


def binarize(img):
    """-> (чернила bool (H, W), серое uint8 с белым фоном и тёмным текстом)."""
    g = to_gray(img)
    sample = g[::5, ::5]
    if np.count_nonzero((sample > 40) & (sample < 215)) < 0.01 * sample.size:
        return g < 128, g                                         # уже чёрно-белое (скан 1 бит и т. п.)
    # фото: неравномерное освещение. Фон = максимум по окну крупнее буквы (на уменьшенной копии), делим на него
    k = max(1, min(g.shape) // 400)
    small = g[::k, ::k].astype(np.float32)
    bg = ndi.maximum_filter(small, size=max(15, min(small.shape) // 25))
    bg = ndi.uniform_filter(bg, size=max(15, min(small.shape) // 25))
    bg = np.asarray(Image.fromarray(bg).resize((g.shape[1], g.shape[0]), Image.BILINEAR))
    norm = np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    return norm < _otsu(norm), norm


def stroke_width(ink):
    """Типичная толщина штриха: медиана горизонтальных серий чернил (поперёк вертикальных стержней)."""
    d = np.diff(np.pad(ink, ((0, 0), (1, 1))).astype(np.int8), axis=1).ravel()
    runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return float(np.median(runs)) if len(runs) else 3.0


def text_stroke_width(ink, lab, area):
    """Толщина штриха с поправкой на шум. Медиана по всем отрезкам (stroke_width) считает каждую пылинку
    наравне с буквой: на фото крупного слова на фоне ткани тысячи точек фактуры дают 4 px при штрихе
    букв в 40 px — и слово целиком уходило в «сплошные пятна» (solid_components) как стол вокруг страницы.
    Поэтому то же самое считается ещё и по крупным компонентам (верхние 10% по площади); если выходит
    вдвое толще — берём это. На страницах и сканах обе оценки совпадают, там ничего не меняется."""
    s = stroke_width(ink)
    if len(area) > 2:
        big = area >= np.percentile(area[1:], 90)
        big[0] = False
        s_big = stroke_width(big[lab] & ink)
        if s_big >= 2 * s:
            return s_big
    return s


def speck_area(area, s):
    """Порог пыли. Точки у букв и знаков препинания в разных шрифтах от 0.15 до 0.8 s² (жирный шрифт — мелкие точки
    относительно штриха), поэтому порог — треть типичной точки на этой же картинке, а не доля от толщины штриха."""
    dots = area[(area >= 0.1 * s * s) & (area <= 1.5 * s * s)]
    if len(dots) >= 5:
        return max(3.0, 0.35 * float(np.median(dots)))
    return max(3.0, 0.08 * s * s)


def solid_components(lab, ink, s):
    """Метки компонент, которые не буквы, а сплошные пятна: стол или тень вокруг
    сфотографированной страницы, чёрная полоса скана, линейка. У буквы любой
    горизонтальный отрезок — это поперечник штриха (≈ s), у пятна отрезки во всю
    его ширину. Без этого тёмный стол вокруг страницы становился одной огромной
    «буквой», и вся страница сливалась в один столбец."""
    starts = ink & ~np.pad(ink, ((0, 0), (1, 0)))[:, :-1]         # начало каждого горизонтального отрезка
    runs = np.bincount(lab[starts], minlength=lab.max() + 1)
    area = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    mean_run = area / np.maximum(runs, 1)
    solid = (mean_run > 5 * s) & (area > 20 * s * s)
    solid[0] = False
    # Но толщину s задаёт самый частый шрифт: на афише рядом с мелкой кириллицей крупная каллиграфия
    # тодо бичиг — «пятно» по этому правилу. Пятна, ради которых оно заведено, либо касаются края
    # кадра, либо тянутся через большую его часть, либо залиты сплошь (отрезок во всю ширину);
    # слово — ни то, ни другое, ни третье.
    H, W = lab.shape
    objs = ndi.find_objects(lab)
    for i in np.flatnonzero(solid):
        sy, sx = objs[i - 1]
        h, w = sy.stop - sy.start, sx.stop - sx.start
        edge = sy.start == 0 or sx.start == 0 or sy.stop == H or sx.stop == W
        wide = h >= 0.6 * H or w >= 0.6 * W
        filled = mean_run[i] >= 0.5 * w
        if not (edge or wide or filled):
            solid[i] = False
    return solid


# ---------------------------------------------------------------- наклон

def _sharpness(ink_small, angle):
    im = Image.fromarray(ink_small.astype(np.uint8) * 255).rotate(angle, resample=Image.BILINEAR, expand=True)
    p = np.asarray(im, dtype=np.float32).sum(0)
    return float((np.diff(p) ** 2).sum())


def estimate_skew(ink, max_angle=5.0):
    """Угол (градусы, для PIL.rotate), выпрямляющий столбцы: у ровных столбцов проекция на ось x самая контрастная."""
    k = max(1, round(max(ink.shape) / 1000))
    small = ink[::k, ::k] if k > 1 else ink
    best = max(np.arange(-max_angle, max_angle + 1e-6, 0.5), key=lambda a: _sharpness(small, a))
    return max(np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1), key=lambda a: _sharpness(small, a))


# ---------------------------------------------------------------- столбцы

def _has_spine(ink, x, s):
    """Есть ли у x сплошная вертикаль чернил длиннее трёх толщин штриха."""
    r = max(1, int(round(s)))
    rows = ink[:, max(0, x - r):x + r + 1].any(1)
    d = np.diff(np.concatenate([[0], rows.astype(np.int8), [0]]))
    runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return bool(len(runs)) and runs.max() >= 3 * s


def _spine_profile(ink, s):
    """Сколько в каждом x пикселей из вертикальных серий чернил длиннее трёх штрихов (стержни, края)."""
    col = np.pad(ink, ((1, 1), (0, 0))).astype(np.int8)
    d = np.diff(col, axis=0)
    out = np.zeros(ink.shape[1], np.float32)
    ys, xs = np.nonzero(d == 1)
    ye, xe = np.nonzero(d == -1)                                  # в каждом x начала и концы идут парами по порядку
    o1, o2 = np.lexsort((ys, xs)), np.lexsort((ye, xe))
    L = ye[o2] - ys[o1]
    long = L >= 3 * s
    np.add.at(out, xs[o1][long], L[long])
    return out


def find_cuts(ink, s):
    """x-координаты границ между столбцами (по минимумам проекции между стержнями)."""
    prof = ndi.uniform_filter1d(ink.sum(0).astype(np.float32), size=max(3, int(round(s))))
    if prof.max() <= 0:
        return []
    top = np.percentile(prof[prof > 0], 95)
    # Высота пика растёт с длиной столбца, поэтому один порог от самого
    # высокого столбца терял короткие: последний столбец из одного слова в 5–10
    # раз ниже остальных и сливался с соседом. Поэтому пик — столбец, если он
    #   • высокий (как раньше), или
    #   • хорошо отделён от соседей (провал рядом глубже половины его высоты)
    #     и под ним есть стержень — сплошная вертикаль длиннее трёх штрихов.
    # Без стержня отделённый пик — это столбик точек или знаков сбоку от оси
    # (в шрифте zakaa они справа), а не столбец.
    peaks, props = find_peaks(prof, prominence=0.02 * top, height=0.05 * top, distance=max(2, int(3 * s)))
    tall = (props["prominences"] >= 0.3 * top) & (props["peak_heights"] >= 0.25 * top)
    apart = props["prominences"] >= 0.5 * props["peak_heights"]
    peaks = peaks[tall | (apart & np.array([_has_spine(ink, x, s) for x in peaks], bool))]
    cuts, spines = [], None
    for a, b in zip(peaks[:-1], peaks[1:]):
        x = a + int(np.argmin(prof[a:b]))
        if prof[x] <= 0.3 * min(prof[a], prof[b]):              # настоящий промежуток, а не провал внутри столбца
            cuts.append(x)
            continue
        # На фото поперёк кадра идут тонкие линии (края крыши, перила), и промежуток до нуля не падает;
        # а мелкий пик рядом (угол здания) делает провал «неглубоким» относительно себя. Но полоса шире
        # трёх штрихов совсем без стержней — вертикальных серий чернил длиннее трёх штрихов — не столбец:
        # у столбца тодо бичиг стержень идёт через всю высоту. Режем в самом пустом месте такой полосы.
        if spines is None:
            spines = _spine_profile(ink, s)
        low = np.concatenate([[0], (spines[a:b] == 0).astype(np.int8), [0]])
        d = np.diff(low)
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts):
            i = int(np.argmax(ends - starts))
            if ends[i] - starts[i] >= 3 * s:
                x0, x1 = a + starts[i], a + ends[i]
                cuts.append(x0 + int(np.argmin(prof[x0:x1])))
    return cuts


def assign_components(lab, cuts, split_share=0.25):
    """Раздаёт компоненты столбцам. -> список: на столбец — список (срез, маска) кусков."""
    edges = np.array([0] + list(cuts) + [lab.shape[1]])
    cols = [[] for _ in range(len(edges) - 1)]
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        xs = np.arange(sl[1].start, sl[1].stop)
        zone = np.searchsorted(edges, xs, side="right") - 1
        mass = np.bincount(zone, weights=m.sum(0), minlength=len(cols))
        order = np.argsort(mass)[::-1]
        if len(order) > 1 and mass[order[1]] >= split_share * mass.sum():   # слиплась с соседним столбцом: режем по границе
            for z in np.flatnonzero(mass):
                part = m & (zone == z)[None, :]
                if part.any():
                    cols[z].append((sl, part))
        else:
            cols[order[0]].append((sl, m))
    return cols


def _bbox(pieces):
    y0 = min(sl[0].start for sl, _ in pieces); y1 = max(sl[0].stop for sl, _ in pieces)
    x0 = min(sl[1].start for sl, _ in pieces); x1 = max(sl[1].stop for sl, _ in pieces)
    return y0, y1, x0, x1


def _tight(sl, m):
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    return (slice(sl[0].start + ys[0], sl[0].start + ys[-1] + 1), slice(sl[1].start + xs[0], sl[1].start + xs[-1] + 1)), \
        m[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]


def column_width(cols_pieces, s):
    """Типичная толщина столбца (по чернилам крупных кусков)."""
    widths = []
    for pieces in cols_pieces:
        big = [p for p in pieces if p[1].sum() > 4 * s * s]
        if big:
            _, _, x0, x1 = _bbox(big)
            widths.append(x1 - x0)
    return float(np.median(widths)) if widths else None


def margin_rows(ink, w, s):
    """Строки колонтитула, заголовка, номера страницы: полоса над или под основным текстом, отделённая от него
    совсем пустыми строками, низкая (строка кириллицы или цифр ниже слова тодо бичиг) и широкая (идёт поперёк
    нескольких столбцов). Знак препинания в конце столбца узкий, а рядом с ним стоят другие столбцы — не трогаем."""
    rows = ink.sum(1)
    filled = rows > s                                             # строка с чем-то, кроме редкой пыли
    d = np.diff(np.concatenate([[0], filled.astype(np.int8), [0]]))
    bands = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    drop = np.zeros(len(rows), bool)

    def is_line(y0, y1):
        xs = np.flatnonzero(ink[y0:y1].any(0))
        return y1 - y0 < 0.75 * w and len(xs) and xs[-1] - xs[0] >= 1.5 * w

    tall = [k for k, (y0, y1) in enumerate(bands) if y1 - y0 >= 0.75 * w]
    if not tall:
        return drop
    for k in list(range(tall[0])) + list(range(tall[-1] + 1, len(bands))):
        y0, y1 = bands[k]
        if is_line(y0, y1):
            drop[y0:y1] = True
    return drop


def drop_margins(cols_pieces, ink, s):
    w = column_width(cols_pieces, s)
    if w is None or sum(1 for c in cols_pieces if c) < 4:        # колонтитулы бывают у страницы; в блоке из 2–3
        return cols_pieces                                        # столбцов правило ошибается чаще, чем помогает
    drop = margin_rows(ink, w, s)
    return [[p for p in pieces if not drop[(p[0][0].start + p[0][0].stop) // 2]] for pieces in cols_pieces]


def _groups(pieces, gap):
    """Куски столбца сверху вниз -> группы, разделённые промежутком больше gap."""
    pieces = sorted(pieces, key=lambda p: p[0][0].start)
    groups, cur, end = [], [pieces[0]], pieces[0][0][0].stop
    for p in pieces[1:]:
        if p[0][0].start - end > gap:
            groups.append(cur); cur = []
        cur.append(p); end = max(end, p[0][0].stop)
    groups.append(cur)
    return groups


def drop_strays(cols_pieces, s):
    """Клочки, далеко оторванные от текста столбца (пыль, одинокий номер страницы), и мелочь в промежутке между
    столбцами: иначе они раздвинули бы рамку столбца."""
    w = column_width(cols_pieces, s)
    if w is None:
        return cols_pieces
    out = []
    for pieces in cols_pieces:
        if not pieces:
            out.append([]); continue
        groups = _groups(pieces, 1.2 * w)                         # промежуток между словами меньше толщины столбца
        ink = [sum(int(m.sum()) for _, m in g) for g in groups]
        main = int(np.argmax(ink))
        keep = [p for k, g in enumerate(groups) if k == main or ink[k] >= 0.15 * ink[main] or ink[k] > 60 * s * s
                for p in g]
        big = [p for p in keep if p[1].sum() > 4 * s * s]
        if big:
            _, _, bx0, bx1 = _bbox(big)
            keep = [p for p in keep if p[1].sum() > 4 * s * s or (bx0 - 0.5 * w <= (p[0][1].start + p[0][1].stop) / 2 <= bx1 + 0.5 * w)]
        out.append(keep)
    return out


# Бледные пиксели, связанные со штрихами столбца, тоже его: порог «бледного» — эта доля пути от порога Оцу
# столбца до белого, дотягиваемся не дальше FAINT_REACH × толщина штриха от уверенных чернил.
FAINT = 0.9
FAINT_REACH = 2


def render_column(pieces, gray, pad, s):
    """Столбец из своих кусков: серые пиксели исходника под маской столбца, всё остальное белое; поля pad ×
    толщина, как в split_columns.

    Маска — чернила столбца, расширенные на 2 px (не срезать сглаженную кромку — для жирных шрифтов это важно),
    и бледные штрихи, которые от них отходят. У рукописи и литографии соединение букв бывает волосной линией
    светлее порога бинаризации: без неё слово на картинке для модели разорвано, и она ставит пробел посреди
    слова (yos bi, kir stos). Расширяемся только по бледным пикселям и недалеко, поэтому соседний столбец не
    затягивается. Настоящие строки test: CER 5.0% -> 3.7%, точных строк 39% -> 51%; синтетика без изменений,
    у Позднеева поменялось 7 столбцов из 180."""
    y0, y1, x0, x1 = _bbox(pieces)
    mask = np.zeros((y1 - y0, x1 - x0), bool)
    for sl, m in pieces:
        mask[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] |= m
    p = round(pad * (x1 - x0))
    q = p + 2
    mask = ndi.binary_dilation(np.pad(mask, q), iterations=2)
    H, W = gray.shape
    crop = np.full(mask.shape, 255, np.uint8)
    gy0, gy1, gx0, gx1 = max(0, y0 - q), min(H, y1 + q), max(0, x0 - q), min(W, x1 + q)
    crop[gy0 - (y0 - q):gy1 - (y0 - q), gx0 - (x0 - q):gx1 - (x0 - q)] = gray[gy0:gy1, gx0:gx1]
    t = _otsu(crop[mask])
    faint = crop < t + (255 - t) * FAINT
    mask = ndi.binary_dilation(mask, iterations=max(1, round(FAINT_REACH * s)), mask=mask | faint)
    out = np.where(mask, crop, 255).astype(np.uint8)[2:-2, 2:-2]
    return Image.fromarray(out), (x0 - p, y0 - p, x1 + p, y1 + p)


def _disk(r):
    return np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r


def _close(mask, r):
    """Замыкание кругом радиуса r через преобразование расстояний: то же, что binary_closing с _disk(r),
    но за линейное время — у большого круга binary_closing на кадре идёт секундами."""
    grown = ndi.distance_transform_edt(~np.pad(mask, r)) <= r
    return (ndi.distance_transform_edt(grown) > r)[r:-r, r:-r]


def find_plates(img, max_plates=6):
    """Надпись на предмете: кулон, табличка, татуировка на руке, обложка на столе. -> [(картинка, рамка в img) ...].

    split_page считает, что кадр — это страница. Когда надпись — лишь часть кадра, а фон пёстрый (ткань,
    кожа дивана, цепочка), фон распадается на «чернила», полярность выбирается по нему, а сама поверхность
    с надписью становится одним сплошным пятном с дырками-буквами — и выкидывается как стол вокруг страницы.

    Здесь поверхность ищут именно как такое пятно:
      1. глобальный порог Оцу по яркости, посчитанный на размытой медианой копии (мелкая фактура не
         сдвигает порог); не binarize — её деление на местный фон превращает тени на коже в дырки;
      2. обе полярности, размыкание кругом ~1/60 кадра — отрываются тонкие перемычки и фактура;
      3. у пятна закрываются дырки (после замыкания кругом в 1/6 пятна — иначе буква у края,
         открытая наружу, дыркой не считается); дырки — кандидаты в надпись, их должно быть
         от 1 до 50% площади;
      4. вырезается рамка вокруг дырок с запасом, всё вне пятна заливается его цветом, чтобы край
         поверхности и фон не стали штрихами.
    Лишнее (складки, соседний кусок фона) отсеивает уже модель по уверенности — см. core/ocr.py."""
    rgb = img.convert("RGB")
    k = min(1.0, 600 / max(rgb.size))
    small = rgb.resize((max(1, round(rgb.width * k)), max(1, round(rgb.height * k))), Image.BILINEAR)
    g = np.asarray(small.convert("L"))
    r = max(2, round(min(g.shape) / 60))
    t = _otsu(ndi.median_filter(g, size=2 * r + 1))
    found = []
    for mask in (g > t, g <= t):
        lab, _ = ndi.label(ndi.binary_opening(mask, _disk(r)))
        area = np.bincount(lab.ravel())
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None or not 0.005 * lab.size <= area[i] <= 0.7 * lab.size:
                continue
            comp = lab[sl] == i
            R = max(2, round(min(comp.shape) / 6))
            closed = ndi.binary_fill_holes(_close(comp, R)) | comp
            hl, _ = ndi.label(closed & ~comp)
            ha = np.bincount(hl.ravel())
            ha[0] = 0
            if not 0.01 * closed.sum() <= ha.sum() <= 0.5 * closed.sum():
                continue
            ys, xs = np.nonzero((ha >= max(4, 0.02 * ha.max()))[hl])   # крошки не раздвигают рамку
            found.append((int(ha.sum()), sl, closed, (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)))
    arr = np.asarray(rgb)
    out = []
    for _, sl, closed, (hy0, hy1, hx0, hx1) in sorted(found, key=lambda f: -f[0])[:max_plates]:
        mg = round(0.15 * max(hx1 - hx0, (hy1 - hy0) / 8))
        hy0, hx0 = max(0, hy0 - mg), max(0, hx0 - mg)
        hy1, hx1 = min(closed.shape[0], hy1 + mg), min(closed.shape[1], hx1 + mg)
        oy, ox = sl[0].start, sl[1].start
        x0, y0 = round((ox + hx0) / k), round((oy + hy0) / k)
        x1, y1 = min(rgb.width, round((ox + hx1) / k)), min(rgb.height, round((oy + hy1) / k))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        # маска поверхности в полном разрешении, чуть уже — чтобы её кромка не стала штрихом
        m = np.asarray(Image.fromarray(closed[hy0:hy1, hx0:hx1].astype(np.uint8) * 255)
                       .resize((x1 - x0, y1 - y0), Image.BILINEAR)) > 127
        m = ndi.binary_erosion(m, iterations=max(1, round(0.03 * min(m.shape))))
        if not m.any():
            continue
        crop = arr[y0:y1, x0:x1]
        fill = np.median(crop[m], axis=0).astype(np.uint8)
        out.append((Image.fromarray(np.where(m[..., None], crop, fill)), (x0, y0, x1, y1)))
    return out


def horizontal_text(lab, n):
    """-> bool по меткам: компоненты, из которых сложены горизонтальные строки — кириллица, латиница, цифры
    на афише, вывеске, обложке рядом с надписью тодо бичиг. Такие строки давали свои пики в проекции,
    сбивали толщину штриха (у мелкого шрифта он тоньше) и срастались со столбцами тодо в один «столбец».

    Буква горизонтального письма — отдельное пятно, не вытянутое вверх; буквы одного кегля стоят рядом
    в длинный ряд. Слово тодо бичиг — одно высокое пятно, а его точки и знаки идут друг под другом.
    Поэтому: пятна не выше полутора ширин разбиваются по классам высоты (с перекрытием, чтобы строка из
    заглавных и строчных не рвалась), в каждом классе расширяются вбок на свою высоту; ряд шире шести
    своих высот из хотя бы четырёх пятен — строка."""
    sl = ndi.find_objects(lab)
    h = np.array([0] + [s[0].stop - s[0].start if s else 0 for s in sl]); w = np.array([0] + [s[1].stop - s[1].start if s else 0 for s in sl])
    cand = (h >= 4) & (h <= 1.5 * w + 2)                     # не вытянутые вверх
    out = np.zeros(n + 1, bool)
    if not cand.any():
        return out
    lo = h[cand].min()
    b = 0
    while lo * 2 ** b <= h[cand].max():
        # класс высоты [lo·2^b, lo·2^(b+2)) — перекрываются, чтобы строка из букв разной высоты не рвалась
        sel = cand & (h >= lo * 2 ** b) & (h < lo * 2 ** (b + 2))
        b += 1
        if sel.sum() < 4:
            continue
        hc = float(np.median(h[sel]))
        m = sel[lab]
        k = max(2, int(round(1.0 * hc)))
        rl, _ = ndi.label(ndi.binary_dilation(m, structure=np.ones((1, 2 * k + 1), bool)))
        for j, s_ in enumerate(ndi.find_objects(rl), 1):
            if s_ is None:
                continue
            rh, rw = s_[0].stop - s_[0].start, s_[1].stop - s_[1].start
            comps = np.unique(lab[s_][(rl[s_] == j) & m[s_]])
            comps = comps[comps > 0]
            if rw >= 6 * rh and len(comps) >= 4:
                out[comps] = True
    return out

def split_page(img, pad=0.1, deskew=True, debug=False):
    """Страница (PIL) -> [столбец PIL ...] слева направо. debug=True -> (столбцы, словарь с промежуточными данными)."""
    ink, gray = binarize(img)
    lab, _ = ndi.label(ink, structure=_EIGHT)
    area = np.bincount(lab.ravel())
    s = text_stroke_width(ink, lab, area)
    keep = (area >= speck_area(area[1:], s)) & ~solid_components(lab, ink, s)
    horiz = horizontal_text(lab, len(area) - 1)
    if area[horiz].sum() < 0.5 * area[1:].sum():                 # строки рядом с тодо бичиг (афиша, вывеска);
        keep &= ~horiz                                            # а страницу, где их большинство, не трогаем:
                                                                  # из обрывков русской страницы модель склеит мусор
    ink = keep[lab] & ink                                         # пылинки и сплошные пятна
    angle = 0.0                                                   # на сколько повернули на самом деле
    if deskew:
        estimate = float(estimate_skew(ink))
        if abs(estimate) >= 0.1:
            angle = estimate
            ink = np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).rotate(angle, Image.NEAREST, expand=True)) > 127
            gray = np.asarray(Image.fromarray(gray).rotate(angle, Image.BICUBIC, expand=True, fillcolor=255))
    lab, _ = ndi.label(ink, structure=_EIGHT)
    cuts = find_cuts(ink, s)
    cols = assign_components(lab, cuts)
    cols = [[_tight(sl, m) for sl, m in c if m.any()] for c in cols]
    cols = drop_margins(cols, ink, s)
    cols = drop_strays(cols, s)
    out, boxes = [], []
    for c in cols:
        if not c or sum(int(m.sum()) for _, m in c) < 8 * s * s:
            continue
        im, box = render_column(c, gray, pad, s)
        out.append(im); boxes.append(box)
    if debug:
        return out, dict(ink=ink, gray=gray, angle=angle, stroke=s, cuts=cuts, boxes=boxes)
    return out
