# -*- coding: utf-8 -*-
"""
page_layout.py — разметка страницы с вертикальным письмом: скан или фото ->
столбцы, которые по одному читает модель распознавания (core/ocr.py).

Модель обучена на отдельных столбцах. Первая версия нарезки (split_columns
из архива модели) искала между столбцами совершенно пустые полосы шириной
от 20 px. На настоящей странице их нет: хвосты букв заходят под соседний
столбец, между столбцами пыль скана, колонтитул и номер страницы идут
поперёк — и вся страница уходила в модель одним «столбцом». Здесь иначе:

1. бинаризация (для фото — с выравниванием фона); пыль — компоненты меньше
   трети типичной точки на этой же картинке;
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
   его компонент, остальное белое, поля pad × толщина — как в рамках, на
   которых училась модель.

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


def binarize(img):
    """-> (чернила bool (H, W), серое uint8 с белым фоном и тёмным текстом)."""
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    border = np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()])
    if np.median(border) < 128:                                   # светлый текст на тёмном фоне
        g = 255 - g
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


def speck_area(area, s):
    """Порог пыли. Точки у букв и знаков препинания в разных шрифтах от 0.15 до 0.8 s² (жирный шрифт — мелкие точки
    относительно штриха), поэтому порог — треть типичной точки на этой же картинке, а не доля от толщины штриха."""
    dots = area[(area >= 0.1 * s * s) & (area <= 1.5 * s * s)]
    if len(dots) >= 5:
        return max(3.0, 0.35 * float(np.median(dots)))
    return max(3.0, 0.08 * s * s)


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

def find_cuts(ink, s):
    """x-координаты границ между столбцами (по минимумам проекции между стержнями)."""
    prof = ndi.uniform_filter1d(ink.sum(0).astype(np.float32), size=max(3, int(round(s))))
    if prof.max() <= 0:
        return []
    top = np.percentile(prof[prof > 0], 95)
    peaks, _ = find_peaks(prof, prominence=0.3 * top, height=0.25 * top, distance=max(2, int(3 * s)))
    cuts = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        x = a + int(np.argmin(prof[a:b]))
        if prof[x] <= 0.3 * min(prof[a], prof[b]):              # настоящий промежуток, а не провал внутри столбца
            cuts.append(x)
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


def render_column(pieces, gray, pad):
    """Столбец из своих кусков: серые пиксели исходника под маской столбца (расширенной на 2 px, чтобы не срезать
    сглаженную кромку — для жирных шрифтов это важно), всё остальное белое; поля pad × толщина, как в split_columns."""
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
    out = np.where(mask, crop, 255).astype(np.uint8)[2:-2, 2:-2]
    return Image.fromarray(out), (x0 - p, y0 - p, x1 + p, y1 + p)


def split_page(img, pad=0.1, deskew=True, debug=False):
    """Страница (PIL) -> [столбец PIL ...] слева направо. debug=True -> (столбцы, словарь с промежуточными данными)."""
    ink, gray = binarize(img)
    s = stroke_width(ink)
    lab, _ = ndi.label(ink, structure=_EIGHT)
    area = np.bincount(lab.ravel())
    ink = (area >= speck_area(area[1:], s))[lab] & ink            # пылинки
    angle = 0.0
    if deskew:
        angle = float(estimate_skew(ink))
        if abs(angle) >= 0.1:
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
        im, box = render_column(c, gray, pad)
        out.append(im); boxes.append(box)
    if debug:
        return out, dict(ink=ink, gray=gray, angle=angle, stroke=s, cuts=cuts, boxes=boxes)
    return out
