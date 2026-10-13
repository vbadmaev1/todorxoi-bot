# -*- coding: utf-8 -*-
"""
ru_translit.py — русский текст -> транслитерация проекта по простым правилам.

Когда language_check узнаёт русский текст (или русское имя), модель калмыцкой
транслитерации ему не нужна: она учена на калмыцких словах и «исправляет»
русские под калмыцкую орфографию (Джамал -> что угодно, только не ǰamal).
Вместо неё — правила чтения по звучанию, на выходе та же латиница проекта
(ǰ č š γ), так что дальше работают те же translit_to_todo и шрифты.

  дж -> ǰ, ж -> ǰ (в тодо бичиг нет отдельной ž), ч -> č, ш -> š, щ -> š,
  ц -> ts, х -> x, г -> g, й -> y, ы -> i, э -> e;
  е ё ю я: в начале слова и после гласной — ye yo yu ya; после согласной
  е -> e, остальные — с йотом (Петя -> petya, Люба -> lyuba);
  ъ и ь не пишутся.

Правила не знают ударений и редукции: Москва -> moskva, а не maskva. Это
запись по буквам, а не по произношению.
"""

import re

_VOWELS = set("аеёиоуыэюя")
_CYR_WORD = re.compile(r"[а-яё]+(?:-[а-яё]+)*", re.IGNORECASE)

_PLAIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "з": "z", "и": "i",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "č",
    "ш": "š", "щ": "š", "ы": "i", "э": "e", "ж": "ǰ", "й": "y",
}
_YOT = {"е": ("ye", "e"), "ё": ("yo", "yo"), "ю": ("yu", "yu"), "я": ("ya", "ya")}


def _word(w: str) -> str:
    w = w.lower()
    out = []
    prev = ""  # предыдущая буква исходного слова
    i = 0
    while i < len(w):
        ch = w[i]
        if ch == "д" and w[i + 1 : i + 2] == "ж":
            out.append("ǰ")
            prev = "ж"
            i += 2
            continue
        if ch in _YOT:
            initial, after_cons = _YOT[ch]
            after_vowel = prev == "" or prev in _VOWELS or prev in "ъь"
            out.append(initial if after_vowel else after_cons)
        elif ch in "ъь":
            pass
        else:
            out.append(_PLAIN.get(ch, ch))
        prev = ch
        i += 1
    return "".join(out)


def russian_to_translit(text: str) -> str:
    """Кириллица -> транслитерация. Всё, кроме слов (знаки, цифры, пробелы,
    латиница), остаётся как есть."""
    return _CYR_WORD.sub(lambda m: "-".join(_word(p) for p in m.group().split("-")), text)
