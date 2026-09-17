# -*- coding: utf-8 -*-
"""
add_marks_to_clear_script.py — дорисовать в шрифт Clear Script знаки
препинания тодо бичиг, которых в нём нет.

ЗАЧЕМ ИМЕННО ТАК

В присланном Clear Script только кириллица, латиница и цифры: ни бирги,
ни запятой, ни четырёх точек. Рисовать их поверх картинки отдельным кодом
значило бы держать вторую систему координат и самому считать переносы.
Проще добавить их в шрифт настоящими глифами — тогда всё остальное
(раскладка, метрики, перенос по столбцам, масштабирование) работает само,
как для любой буквы.

ОТКУДА ГЕОМЕТРИЯ

Формы взяты не на глаз: замерены пропорции тех же знаков в
MongolianUniversalWhite и пересчитаны под Clear Script. Разница в том,
где проходит стержень — линия, к которой лепятся буквы: у Universal она
на y=440, у Clear Script на y=355 (в единицах шрифта, 1000 на em).
Знаки ставятся относительно стержня, а не базовой линии, иначе они
поедут вбок.

Ромбы рисуются как есть, тире — прямоугольником (вертикальным оно
становится после поворота строки), бирга собрана из дуг.

Запуск (нужен только при замене шрифта):

    python -m tools.add_marks_to_clear_script \\
        --src assets/clear_script.ttf --out assets/clear_script.ttf
"""

import argparse
import math
from pathlib import Path

from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

# --- где проходит стержень в каждом шрифте (единицы, 1000/em) ---------------
SPINE_CLEAR = 355
SPINE_REF = 440          # MongolianUniversalWhite, для пересчёта

# --- размеры ромба, снятые с эталона ---------------------------------------
RHOMB_HW = 56            # половина ширины (вдоль чтения)
RHOMB_HH = 127           # половина высоты (поперёк столбца)
BELOW_SPINE = 55         # насколько центр знака ниже стержня

# --- расстановка ------------------------------------------------------------
STOP_GAP = 255           # между двумя ромбами точки
FOUR_DX = 122            # разнос четырёх точек вдоль чтения
FOUR_DY = 175            # и поперёк
DASH_LEN = 918
DASH_THICK = 59

MARKS = {
    0x1800: "uni1800",   # ᠀ бирга
    0x1802: "uni1802",   # ᠂ запятая
    0x1803: "uni1803",   # ᠃ точка
    0x1805: "uni1805",   # ᠅ четыре точки
    0xFE31: "uniFE31",   # ︱ тире
    0x202F: "uni202F",   # узкий неразрывный пробел — граница суффикса
}

# Пустой глиф: рисовать нечего, важен только шаг. У обычного пробела в
# Clear Script шаг 160; узкий делаем чуть уже, в той же пропорции, что и
# в эталонном шрифте (244 против 277, то есть примерно 0.88).
NARROW_SPACE_ADVANCE = 140


def _cy():
    """Центр знака по вертикали: ниже стержня, как в эталоне."""
    return SPINE_CLEAR - BELOW_SPINE


def _rhomb(pen, cx, cy, hw=RHOMB_HW, hh=RHOMB_HH):
    pen.moveTo((cx - hw, cy))
    pen.lineTo((cx, cy + hh))
    pen.lineTo((cx + hw, cy))
    pen.lineTo((cx, cy - hh))
    pen.closePath()


def _rect(pen, x0, y0, x1, y1):
    pen.moveTo((x0, y0))
    pen.lineTo((x0, y1))
    pen.lineTo((x1, y1))
    pen.lineTo((x1, y0))
    pen.closePath()


def _stroke(pen, path, widths):
    """Каллиграфический штрих: ведём перо вдоль пути, меняя толщину.

    Контур строим сами — сначала левый край вперёд, потом правый назад.
    Так одна незамкнутая линия превращается в замкнутую фигуру, а
    переменная ширина даёт характерный нажим, без которого завиток
    выглядит проволокой.
    """
    left, right = [], []
    n = len(path)
    for i, (x, y) in enumerate(path):
        # направление берём по соседям — на концах по одному соседу
        px, py = path[max(i - 1, 0)]
        nx, ny = path[min(i + 1, n - 1)]
        dx, dy = nx - px, ny - py
        length = math.hypot(dx, dy) or 1.0
        # нормаль к направлению
        ox, oy = -dy / length, dx / length
        half = widths[i] / 2
        left.append((x + ox * half, y + oy * half))
        right.append((x - ox * half, y - oy * half))
    pen.moveTo(left[0])
    for p in left[1:]:
        pen.lineTo(p)
    for p in reversed(right):
        pen.lineTo(p)
    pen.closePath()


def _bezier(p0, p1, p2, steps):
    return [
        ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
         (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
        for t in (i / steps for i in range(steps + 1))
    ]


def _spiral(center, r0, r1, a0, turns, steps):
    cx, cy = center
    pts = []
    for i in range(steps + 1):
        t = i / steps
        ang = a0 + turns * 2 * math.pi * t
        r = r0 + (r1 - r0) * t
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def draw_comma(pen):
    _rhomb(pen, 201, _cy())
    return 391


def draw_stop(pen):
    cy = _cy()
    _rhomb(pen, 201, cy)
    _rhomb(pen, 201 + STOP_GAP, cy)
    return 642


def draw_four(pen):
    cx, cy = 279, _cy()
    _rhomb(pen, cx - FOUR_DX, cy)
    _rhomb(pen, cx + FOUR_DX, cy)
    _rhomb(pen, cx, cy + FOUR_DY)
    _rhomb(pen, cx, cy - FOUR_DY)
    return 552


def draw_dash(pen):
    cy = _cy()
    _rect(pen, 171, cy - DASH_THICK / 2, 171 + DASH_LEN, cy + DASH_THICK / 2)
    return 1123


def draw_birga(pen):
    """Бирга — знак начала записи: длинный хвост, уходящий в завиток.

    Пропорции сняты с эталона: по ширине 20..315, по высоте 621 единица,
    шаг 333. Пересчитано относительно стержня Clear Script, поэтому в
    координатах здесь вычтено смещение между стержнями двух шрифтов.
    """
    shift = SPINE_CLEAR - SPINE_REF          # -85: опускаем всю фигуру

    # хвост: от тонкого кончика внизу дугой вверх-вправо
    tail = _bezier((70, 29 + shift), (215, 150 + shift), (249, 499 + shift), 24)
    # завиток: полтора оборота против часовой, радиус сходит на нет
    curl = _spiral((152, 472 + shift), 116, 34, math.radians(-42), 1.12, 44)

    path = tail + curl[1:]
    # нажим: тонко на кончике хвоста, полно в середине, тоньше к центру завитка
    widths = ([14 + 44 * (i / (len(tail) - 1)) ** 0.7 for i in range(len(tail))]
              + [58 - 40 * (i / (len(curl) - 2)) for i in range(len(curl) - 1)])
    _stroke(pen, path, widths)
    return 333


def draw_narrow_space(pen):
    """Ничего не рисуем — нужен только шаг."""
    return NARROW_SPACE_ADVANCE


BUILDERS = {
    0x202F: draw_narrow_space,
    0x1800: draw_birga,
    0x1802: draw_comma,
    0x1803: draw_stop,
    0x1805: draw_four,
    0xFE31: draw_dash,
}


def add_marks(src: Path, out: Path) -> None:
    font = TTFont(src)
    upem = font["head"].unitsPerEm
    scale = upem / 1000.0
    glyf, hmtx = font["glyf"], font["hmtx"]
    order = font.getGlyphOrder()

    added = []
    for cp, name in MARKS.items():
        pen = TTGlyphPen(None)
        advance = BUILDERS[cp](pen)
        glyph = pen.glyph()
        if scale != 1.0:  # на случай другого upem
            glyph.recalcBounds(glyf)
        glyf[name] = glyph
        hmtx[name] = (int(advance * scale), 0)
        if name not in order:
            order.append(name)
        added.append(name)

    font.setGlyphOrder(order)
    font["maxp"].numGlyphs = len(order)
    # в шрифте есть и однобайтовая подтаблица (format 0) — в неё коды выше
    # 255 просто не влезают, поэтому пишем только в те, что их принимают
    for table in font["cmap"].tables:
        if table.format == 0:
            continue
        for cp, name in MARKS.items():
            table.cmap[cp] = name

    out.parent.mkdir(parents=True, exist_ok=True)
    font.save(out)
    print(f"Добавлено глифов: {len(added)} -> {out}")
    for cp, name in MARKS.items():
        g = font["glyf"][name]
        step = font["hmtx"][name][0]
        if g.numberOfContours == 0:   # пустой глиф — у него нет границ
            print(f"  U+{cp:04X} (пусто)  шаг={step}")
        else:
            print(f"  U+{cp:04X} {chr(cp)}  x:{g.xMin}..{g.xMax}"
                  f"  y:{g.yMin}..{g.yMax}  шаг={step}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("assets/clear_script.ttf"))
    ap.add_argument("--out", type=Path, default=Path("assets/clear_script.ttf"))
    args = ap.parse_args()
    add_marks(args.src, args.out)


if __name__ == "__main__":
    main()
