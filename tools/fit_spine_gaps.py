# -*- coding: utf-8 -*-
"""
fit_spine_gaps.py — где в кириллической записи не хватает стержня.

ПРОБЛЕМА

Шрифты семейства Clear Script соединяют буквы перекрытием: каждая рисует
свой кусок стержня и заходит на соседнюю. Обычно это и даёт непрерывную
линию. Но у некоторых букв чернила уходят далеко влево от начала — и
тогда предыдущая буква заливает их просвет. Самый заметный случай:

    Тиим -> teyimü -> тэйимю — хвост «м» закрашивает петлю «ю».

В самом шрифте для этого есть «_»: короткий отрезок стержня. Он удлиняет
спину слова и разводит буквы, не разрывая линию.

КАК РЕШАЕМ, ЧТО ПАРЕ НУЖЕН ОТРЕЗОК

Не по перекрытию прямоугольников: в курсивном письме буквы обязаны
перекрываться, и по габаритам нормальное соединение неотличимо от
испорченного. Смотрим на то, что портится на самом деле, — на просветы.

Растеризуем каждую букву пары отдельно, у второй ищем замкнутые белые
области (петли, очко буквы) и считаем, какую их долю закрашивает первая.
Потом то же в обратную сторону. Если закрашено больше FILL_LIMIT —
подставляем «_» и проверяем снова, пока не станет чисто (не больше
MAX_MARKS штук).

Мера получается избирательной: из 5041 пары алфавита записи срабатывают
единицы, а обычные соединения (бя, им, ло) дают ровно ноль.

Результат — таблица assets/clear_script_gaps.json: для каждого шрифта
только те пары, которым нужен отрезок, и сколько именно. Её читает
core/clear_script.py.

Запуск (нужен при замене или добавлении шрифта):

    python -m tools.fit_spine_gaps
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from core.clear_script import FONT_FILES, RULES_PATH, font_path
from core.shaper import shape

SIZE = 120          # кегль растеризации: мельче — просветы схлопываются
PAD = 24            # поля холста, чтобы заливка фона обошла буквы кругом
FILL_LIMIT = 0.12   # какую долю просвета согласны потерять
MAX_MARKS = 3       # больше — уже не соединение, а разрыв слова
SPINE_MARK = "_"

GAPS_PATH = RULES_PATH.parent / "clear_script_gaps.json"


def _layers(text: str, font_path_str: str):
    """Маски отдельных глифов строки, выставленные на общий холст."""
    run = shape(text, font_path_str, SIZE)
    size = (run.ink_width + 2 * PAD, run.ink_height + 2 * PAD)
    out = []
    for _gid, x, y, bitmap in run.glyphs:
        canvas = Image.new("L", size, 0)
        canvas.paste(bitmap, (x - run.ink_left + PAD, y - run.ink_top + PAD))
        out.append(canvas.point(lambda v: 255 if v > 128 else 0))
    return out


def _holes(mask: Image.Image) -> Image.Image:
    """Замкнутые просветы буквы: заливаем фон от угла, что осталось белым
    внутри чернил — и есть петли."""
    flooded = mask.copy()
    ImageDraw.floodfill(flooded, (0, 0), 128)
    # 0 — не закрашенный заливкой фон, то есть внутренние просветы
    return flooded.point(lambda v: 255 if v == 0 else 0)


def _filled(hole_owner: Image.Image, other: Image.Image) -> float:
    """Какая доля просветов одной буквы закрашена чернилами другой."""
    holes = _holes(hole_owner)
    total = sum(holes.histogram()[128:])
    if not total:
        return 0.0
    both = Image.new("L", holes.size, 0)
    both.paste(other, (0, 0), holes)
    return sum(both.histogram()[128:]) / total


def damage(text: str, font_path_str: str) -> float:
    """Насколько сильно соседние буквы съедают просветы друг друга."""
    layers = _layers(text, font_path_str)
    if len(layers) < 2:
        return 0.0
    first, last = layers[0], layers[-1]
    rest_before = Image.new("L", first.size, 0)
    for layer in layers[:-1]:
        rest_before.paste(layer, (0, 0), layer)
    rest_after = Image.new("L", first.size, 0)
    for layer in layers[1:]:
        rest_after.paste(layer, (0, 0), layer)
    return max(_filled(last, rest_before), _filled(first, rest_after))


def marks_for_pair(pair: str, font_path_str: str) -> int:
    """Сколько отрезков стержня нужно вставить между двумя буквами."""
    if damage(pair, font_path_str) <= FILL_LIMIT:
        return 0
    for n in range(1, MAX_MARKS + 1):
        candidate = pair[0] + SPINE_MARK * n + pair[1]
        if damage(candidate, font_path_str) <= FILL_LIMIT:
            return n
    return MAX_MARKS


def record_alphabet() -> str:
    """Все буквы, которые вообще может выдать запись для шрифта."""
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    letters = set()
    for table in ("r_uni", "r_pos", "r_prev", "r_next", "r_tri"):
        for variants in rules[table].values():
            for chunk in variants:
                letters.update(ch for ch in chunk if ch.isalpha())
    return "".join(sorted(letters))


def fit(keys=None) -> dict:
    alphabet = record_alphabet()
    print(f"алфавит записи: {len(alphabet)} букв, "
          f"{len(alphabet) ** 2} пар на шрифт")
    table = {}
    for key in keys or FONT_FILES:
        path = str(font_path(key))
        found = {}
        for first in alphabet:
            for second in alphabet:
                n = marks_for_pair(first + second, path)
                if n:
                    found[first + second] = n
        table[key] = found
        listing = ", ".join(f"{p}×{n}" for p, n in sorted(found.items()))
        print(f"  {key:9s} пар со стержнем: {len(found):3d}  {listing}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fonts", nargs="*", help="ключи шрифтов, по умолчанию все")
    ap.add_argument("--out", type=Path, default=GAPS_PATH)
    args = ap.parse_args()

    table = fit(args.fonts or None)
    args.out.write_text(
        json.dumps(table, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    print(f"записано: {args.out}")


if __name__ == "__main__":
    main()
