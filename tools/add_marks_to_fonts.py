# -*- coding: utf-8 -*-
"""
add_marks_to_fonts.py — дорисовать знаки препинания тодо бичиг в шрифты
семейства Clear Script.

ЗАЧЕМ ИМЕННО ТАК

В этих шрифтах только кириллица, латиница и цифры: ни бирги, ни запятой,
ни четырёх точек. Рисовать их поверх картинки отдельным кодом значило бы
держать вторую систему координат и самому считать переносы. Проще
добавить их в шрифт настоящими глифами — тогда раскладка, метрики и
перенос по столбцам работают сами, как для любой буквы.

ПОЧЕМУ ВСЁ ЗАМЕРЯЕТСЯ, А НЕ ЗАДАНО ЧИСЛАМИ

Шрифтов пять, и они разные: у Zakaa единиц на em полторы тысячи вместо
тысячи, стержень у каждого проходит на своей высоте, а чернила букв
заходят за шаг по-разному — от 73 единиц у Garcaq до 265 у Zakaa. Поэтому
для каждого шрифта отдельно замеряются:

  * upem       — все координаты ниже даны в тысячных долях em и
                 домножаются на масштаб;
  * стержень   — линия, к которой лепятся буквы. Ищется рендером: в
                 строке из букв это строка пикселей с наибольшим числом
                 чернил. Знаки ставятся относительно неё, иначе поедут;
  * вынос      — насколько чернила букв вылезают за шаг. Буквы
                 соединяются, поэтому перо после слова стоит внутри
                 последней буквы, и знак без отступа налезает на её
                 хвост. Отступ слева берётся как вынос плюс зазор.

Формы взяты не на глаз: пропорции сняты с MongolianUniversalWhite, где
эти знаки есть. Ромбы, тире и бирга рисуются заново, а вопросительный и
восклицательный знаки берутся из самого шрифта — их поворот решается по
самому глифу, см. TAKEN_FROM.

Запуск (нужен при замене или добавлении шрифта):

    python -m tools.add_marks_to_fonts assets/clear_script.ttf ...
    python -m tools.add_marks_to_fonts --all
"""

import argparse
import math
from pathlib import Path

from fontTools.misc.transform import Transform
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

# --- эталон: MongolianUniversalWhite, координаты в тысячных долях em -------
SPINE_REF = 440          # где проходит стержень у эталона
RHOMB_HW = 56            # половина ширины ромба (вдоль чтения)
RHOMB_HH = 127           # половина высоты (поперёк столбца)
BELOW_SPINE = 55         # насколько центр знака ниже стержня
STOP_GAP = 255           # между двумя ромбами точки
FOUR_DX = 122            # разнос четырёх точек вдоль чтения
FOUR_DY = 175            # и поперёк
DASH_LEN = 918
DASH_THICK = 59
BIRGA_ADVANCE = 333

# зазор между хвостом буквы и знаком, сверх замеренного выноса чернил
MARK_GAP = 70

MARKS = {
    0x1800: "uni1800",   # ᠀ бирга
    0x1802: "uni1802",   # ᠂ запятая
    0x1803: "uni1803",   # ᠃ точка
    0x1805: "uni1805",   # ᠅ четыре точки
    0xFE31: "uniFE31",   # ︱ тире
    0x202F: "uni202F",   # узкий неразрывный пробел — граница суффикса
    0xFE15: "uniFE15",   # ︕ восклицательный, повёрнутый
    0xFE16: "uniFE16",   # ︖ вопросительный, повёрнутый
}

# Эти два не рисуем, а берём из собственных глифов шрифта.
#
# Поворачивать или нет — решаем по самому глифу. Шрифты семейства
# расходятся: clear_script, demberil и garcaq рисуют ? и ! стоя, как в
# обычном тексте (они выше, чем шире), а biyir и zakaa — уже лёжа, под
# поворот строки. Если повернуть вторые, знаки лягут набок.
TAKEN_FROM = {0xFE15: "!", 0xFE16: "?"}
ROTATE_DEG = 90          # подобрано рендером: после поворота строки стоят прямо

NARROW_SPACE_RATIO = 0.88   # узкий пробел относительно обычного

SPINE_SAMPLE = "алимнрстуоэ"   # на чём измеряем стержень


# ---------------------------------------------------------------- измерения

def measure_spine(font_path: Path, upem: int) -> int:
    """Где проходит стержень — горизонтальная линия, к которой лепятся
    буквы. Ищем строку пикселей с максимумом чернил и переводим в единицы
    шрифта относительно базовой линии."""
    size = 200
    pil = ImageFont.truetype(str(font_path), size)
    img = Image.new("L", (2000, 700), 0)
    base = (100, 450)
    ImageDraw.Draw(img).text(base, SPINE_SAMPLE, font=pil, fill=255, anchor="ls")
    rows = [sum(img.crop((0, y, img.width, y + 1)).tobytes()) for y in range(img.height)]
    peak = max(range(len(rows)), key=lambda y: rows[y])
    return round((base[1] - peak) / size * upem)


def measure_overhang(font: TTFont) -> int:
    """Насколько сильно чернила букв вылезают за шаг."""
    glyf, hmtx = font["glyf"], font["hmtx"]
    values = [
        glyf[name].xMax - hmtx[name][0]
        for name in font.getGlyphOrder()
        if glyf[name].numberOfContours > 0
    ]
    return max(values) if values else 0


# ------------------------------------------------------------------ фигуры

class MarkBuilder:
    """Рисует знаки под конкретный шрифт: свой масштаб, стержень и отступ."""

    def __init__(self, upem: int, spine: int, left_pad: int):
        self.k = upem / 1000.0        # координаты заданы в тысячных долях em
        self.spine = spine
        self.pad = left_pad
        self.cy = spine - BELOW_SPINE * self.k

    def u(self, value):
        """Единицы эталона -> единицы этого шрифта."""
        return value * self.k

    def _rhomb(self, pen, cx, cy):
        hw, hh = self.u(RHOMB_HW), self.u(RHOMB_HH)
        pen.moveTo((cx - hw, cy))
        pen.lineTo((cx, cy + hh))
        pen.lineTo((cx + hw, cy))
        pen.lineTo((cx, cy - hh))
        pen.closePath()

    def comma(self, pen):
        self._rhomb(pen, self.pad + self.u(RHOMB_HW), self.cy)
        return self.pad + self.u(391 - 145)

    def stop(self, pen):
        x = self.pad + self.u(RHOMB_HW)
        self._rhomb(pen, x, self.cy)
        self._rhomb(pen, x + self.u(STOP_GAP), self.cy)
        return self.pad + self.u(642 - 145)

    def four(self, pen):
        cx = self.pad + self.u(FOUR_DX + RHOMB_HW)
        self._rhomb(pen, cx - self.u(FOUR_DX), self.cy)
        self._rhomb(pen, cx + self.u(FOUR_DX), self.cy)
        self._rhomb(pen, cx, self.cy + self.u(FOUR_DY))
        self._rhomb(pen, cx, self.cy - self.u(FOUR_DY))
        return self.pad + self.u(552 - 101)

    def dash(self, pen):
        half = self.u(DASH_THICK) / 2
        x0 = self.pad
        x1 = x0 + self.u(DASH_LEN)
        pen.moveTo((x0, self.cy - half))
        pen.lineTo((x0, self.cy + half))
        pen.lineTo((x1, self.cy + half))
        pen.lineTo((x1, self.cy - half))
        pen.closePath()
        return x1 + self.pad

    def birga(self, pen):
        """Хвост, уходящий в завиток. Строится штрихом с переменным
        нажимом: ровная линия постоянной толщины выглядит проволокой."""
        shift = self.spine - self.u(SPINE_REF)
        p = lambda x, y: (self.u(x) + self.pad - self.u(20), self.u(y) + shift)

        tail = _bezier(p(70, 29), p(215, 150), p(249, 499), 24)
        curl = _spiral(p(152, 472), self.u(116), self.u(34), math.radians(-42), 1.12, 44)
        path = tail + curl[1:]
        widths = (
            [self.u(14 + 44 * (i / (len(tail) - 1)) ** 0.7) for i in range(len(tail))]
            + [self.u(58 - 40 * (i / (len(curl) - 2))) for i in range(len(curl) - 1)]
        )
        _stroke(pen, path, widths)
        return self.pad + self.u(BIRGA_ADVANCE - 20)


def _stroke(pen, path, widths):
    """Каллиграфический штрих: ведём перо вдоль пути, меняя толщину.
    Контур строим сами — левый край вперёд, правый назад."""
    left, right = [], []
    n = len(path)
    for i, (x, y) in enumerate(path):
        px, py = path[max(i - 1, 0)]
        nx, ny = path[min(i + 1, n - 1)]
        dx, dy = nx - px, ny - py
        length = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / length, dx / length
        half = widths[i] / 2
        left.append((x + ox * half, y + oy * half))
        right.append((x - ox * half, y - oy * half))
    pen.moveTo(left[0])
    for point in left[1:]:
        pen.lineTo(point)
    for point in reversed(right):
        pen.lineTo(point)
    pen.closePath()


def _bezier(p0, p1, p2, steps):
    return [
        ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
         (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
        for t in (i / steps for i in range(steps + 1))
    ]


def _spiral(center, r0, r1, a0, turns, steps):
    cx, cy = center
    out = []
    for i in range(steps + 1):
        t = i / steps
        ang = a0 + turns * 2 * math.pi * t
        r = r0 + (r1 - r0) * t
        out.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return out


def _placed_glyph(font, src_char, left_pad, cy):
    """Собственный глиф шрифта, поставленный на своё место у стержня.

    Поворачиваем только если знак нарисован стоя — то есть выше, чем
    шире. У части шрифтов семейства он уже лежит под поворот строки, и
    второй поворот положил бы его набок.
    """
    glyphset = font.getGlyphSet()
    cmap = font.getBestCmap()
    if ord(src_char) not in cmap:
        return None, 0
    name = cmap[ord(src_char)]
    source = font["glyf"][name]
    if source.numberOfContours == 0:
        return None, 0

    upright = (source.yMax - source.yMin) > (source.xMax - source.xMin)
    angle = math.radians(ROTATE_DEG) if upright else 0.0

    probe = TTGlyphPen(None)
    glyphset[name].draw(TransformPen(probe, Transform().rotate(angle)))
    box = probe.glyph()
    box.recalcBounds(font["glyf"])

    dx = left_pad - box.xMin
    dy = cy - (box.yMin + box.yMax) / 2
    pen = TTGlyphPen(None)
    glyphset[name].draw(
        TransformPen(pen, Transform().translate(dx, dy).rotate(angle))
    )
    glyph = pen.glyph()
    glyph.recalcBounds(font["glyf"])
    return glyph, glyph.xMax + left_pad


# ------------------------------------------------------------------- сборка

def add_marks(path: Path, out: Path = None) -> None:
    out = out or path
    font = TTFont(path)
    upem = font["head"].unitsPerEm
    spine = measure_spine(path, upem)
    overhang = measure_overhang(font)
    left_pad = overhang + round(MARK_GAP * upem / 1000)

    builder = MarkBuilder(upem, spine, left_pad)
    shapes = {
        0x1800: builder.birga,
        0x1802: builder.comma,
        0x1803: builder.stop,
        0x1805: builder.four,
        0xFE31: builder.dash,
    }
    space = font["hmtx"][font.getBestCmap()[0x20]][0]

    glyf, hmtx = font["glyf"], font["hmtx"]
    order = font.getGlyphOrder()
    written = []

    for cp, name in MARKS.items():
        if cp == 0x202F:
            glyph, advance = TTGlyphPen(None).glyph(), space * NARROW_SPACE_RATIO
        elif cp in TAKEN_FROM:
            glyph, advance = _placed_glyph(
                font, TAKEN_FROM[cp], left_pad, builder.cy
            )
            if glyph is None:
                continue
        else:
            pen = TTGlyphPen(None)
            advance = shapes[cp](pen)
            glyph = pen.glyph()
            glyph.recalcBounds(glyf)
        glyf[name] = glyph
        hmtx[name] = (int(round(advance)), 0)
        if name not in order:
            order.append(name)
        written.append(name)

    font.setGlyphOrder(order)
    font["maxp"].numGlyphs = len(order)
    # у этих шрифтов есть однобайтовая подтаблица cmap (format 0) — коды
    # выше 255 в неё не влезают, пишем только в те, что их принимают
    for table in font["cmap"].tables:
        if table.format == 0:
            continue
        for cp, name in MARKS.items():
            if name in glyf:
                table.cmap[cp] = name

    out.parent.mkdir(parents=True, exist_ok=True)
    font.save(out)
    print(f"{path.name}: upem={upem}, стержень={spine}, вынос={overhang}, "
          f"отступ={left_pad} -> добавлено {len(written)} глифов")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fonts", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true",
                    help="все шрифты семейства из assets/")
    args = ap.parse_args()

    paths = list(args.fonts)
    if args.all or not paths:
        assets = Path(__file__).resolve().parent.parent / "assets"
        paths = sorted(p for p in assets.glob("*.ttf")
                       if p.name != "MongolianUniversalWhite.ttf")
    for path in paths:
        add_marks(path)


if __name__ == "__main__":
    main()
