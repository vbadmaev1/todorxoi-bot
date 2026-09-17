# -*- coding: utf-8 -*-
"""
clear_script.py — второй способ нарисовать тодо бичиг: шрифт Clear Script.

ЧЕМ ОН ОТЛИЧАЕТСЯ ОТ ОСНОВНОГО

MongolianUniversalWhite работает с настоящим юникодом тодо бичиг: ему
подают ᡍᠠᠯᡅᡏᠠᡎ, а нужные формы букв выбирает HarfBuzz по таблицам GSUB.

Clear Script устроен иначе: в нём нет ни одной монгольской кодовой точки,
только кириллица, латиница и цифры. Форму буквы задаёт не движок, а сам
набор: строчная и заглавная кириллические буквы — это разные начертания
(начальное, срединное, конечное), и «ХалимаЩ» рисуется как хальмг. То
есть шрифт ждёт особую запись, а не текст.

Эту запись строят правила из assets/clear_script_rules.json: статистика,
собранная выравниванием словаря (транслитерация ↔ запись для шрифта).
Здесь лежит только та половина, что нужна боту, — перевод транслитерации
в запись. Алгоритм: динамическое программирование по сегментации слова на
куски в 1–2 буквы, для каждого куска берётся самый частый вариант с
откатом по контексту (позиция в слове + соседние буквы → позиция →
общий).

ВАЖНО ПРО ТОЧНОСТЬ. Правила статистические: на словах из словаря они
воспроизводят запись почти всегда, на редких и незнакомых — могут
ошибиться в форме буквы. Это цена того, что шрифт не умеет выбирать формы
сам. Основной шрифт такой особенности не имеет, поэтому он и оставлен по
умолчанию.

ЗНАКИ ПРЕПИНАНИЯ. В присланном шрифте их не было вовсе, поэтому бирга,
запятая, точка, четыре точки, тире и узкий неразрывный пробел дорисованы
в него отдельными глифами — см. tools/add_marks_to_clear_script.py. Здесь
мы просто не выбрасываем знаки из текста, а переводим в те же коды, что и
основной путь, чтобы дальше сработала общая логика из punctuation.py.
"""

import json
import math
import re
import threading
import unicodedata
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RULES_PATH = _HERE.parent / "assets" / "clear_script_rules.json"
FONT_PATH = _HERE.parent / "assets" / "clear_script.ttf"

MAX_SPAN = 2        # сколько букв латиницы может лечь в один кусок
MIN_CTX_COUNT = 3   # ниже этого контекстное правило считаем ненадёжным

# долгота выносится отдельным символом: так её удобно выравнивать с
# мягким знаком, которым она записывается в шрифте
_MACRON_SPLIT = {
    "ā": "aˉ", "ē": "eˉ", "ī": "iˉ", "ō": "oˉ", "ū": "uˉ", "ȫ": "öˉ", "ǖ": "üˉ",
}

# Знаки препинания переводим в те же вертикальные формы, что и основной
# путь: дальше их подхватит punctuation.py и превратит в ᠂ ᠃ ᠅.
_PUNCT = {
    ",": "︐",
    ".": "︒",
    "!": "︕",
    "?": "︖",
}

# буквы транслитерации — всё остальное считаем «между словами»
_LETTERS_RE = re.compile(r"[a-zāēīōūȫǖöüγčšǰˉ]+", re.IGNORECASE)


def _split_lat(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).strip().lower()
    s = re.sub(r"\s+", "-", s)
    return "".join(_MACRON_SPLIT.get(c, c) for c in s)


class ClearScriptRules:
    """Транслитерация -> запись для шрифта Clear Script."""

    def __init__(self, path=None):
        data = json.loads(Path(path or RULES_PATH).read_text(encoding="utf-8"))
        self.r_uni = data["r_uni"]
        self.r_pos = data["r_pos"]
        self.r_prev = data["r_prev"]
        self.r_next = data["r_next"]
        self.r_tri = data["r_tri"]
        self.unit_rate = data["unit_rate"]

    @staticmethod
    def _pick(table, key, min_count):
        d = table.get(key)
        return d if d and sum(d.values()) >= min_count else None

    def _variants(self, pos, prev, unit, nxt):
        """Откат по контексту: чем конкретнее правило, тем оно важнее."""
        return (self._pick(self.r_tri, f"{pos}|{prev}|{unit}|{nxt}", MIN_CTX_COUNT)
                or self._pick(self.r_prev, f"{pos}|{prev}|{unit}", MIN_CTX_COUNT)
                or self._pick(self.r_next, f"{pos}|{unit}|{nxt}", MIN_CTX_COUNT)
                or self._pick(self.r_pos, f"{pos}|{unit}", 1)
                or self._pick(self.r_uni, unit, 1))

    def _word(self, s: str) -> str:
        """Одно слово: ищем разбиение на куски с наибольшим суммарным весом."""
        n = len(s)
        best = [(-math.inf, None)] * (n + 1)
        best[0] = (0.0, None)
        for j in range(n):
            if best[j][0] == -math.inf:
                continue
            for k in range(1, MAX_SPAN + 1):
                if j + k > n:
                    break
                unit = s[j:j + k]
                if unit not in self.r_uni:
                    continue
                end = j + k == n
                pos = "S" if (j == 0 and end) else "I" if j == 0 else "F" if end else "M"
                prev = s[j - 1] if j > 0 else "#"
                nxt = s[j + k] if not end else "#"
                variants = self._variants(pos, prev, unit, nxt)
                if not variants:
                    continue
                chunk, count = max(variants.items(), key=lambda kv: kv[1])
                score = (best[j][0]
                         + math.log(self.unit_rate.get(unit, 1e-3) + 1e-9)
                         + math.log(count / sum(variants.values())))
                if score > best[j + k][0]:
                    best[j + k] = (score, (j, chunk))
        if best[n][1] is None:
            return s  # ничего не подобралось — отдаём как есть, видно будет сразу
        chunks, j = [], n
        while j > 0:
            prev_j, chunk = best[j][1]
            chunks.append(chunk)
            j = prev_j
        return "".join(reversed(chunks))

    def translit_to_font(self, translit: str) -> str:
        """Строка транслитерации -> строка, которую рисует Clear Script.

        Буквы идут через правила, всё остальное — знаки препинания, тире,
        пробелы — переносится один в один, только в те же коды, что и на
        основном пути. Поэтому текст остаётся текстом: и перенос по
        столбцам, и расстановка бирги работают как обычно.
        """
        from .translit_todo import _dashes_to_vertical

        out_lines = []
        for line in _dashes_to_vertical(str(translit)).split("\n"):
            out, pos = [], 0
            for m in _LETTERS_RE.finditer(line):
                out.append(self._between(line[pos:m.start()]))
                out.append(self._word(_split_lat(m.group())))
                pos = m.end()
            out.append(self._between(line[pos:]))
            out_lines.append("".join(out))
        return "\n".join(out_lines)

    @staticmethod
    def _between(chunk: str) -> str:
        """То, что между словами: пробелы, знаки, дефис-граница суффикса."""
        out = []
        for ch in chunk:
            if ch in _PUNCT:
                out.append(_PUNCT[ch])
            elif ch == "-":
                out.append("\u202f")      # граница суффикса, как в основном пути
            elif ch.isspace() or ch == "\uFE31":
                out.append(ch)
            # остальное (цифры, скобки) в этом шрифте всё равно не нужно
        return "".join(out)


_RULES = None
_LOCK = threading.Lock()


def get_rules() -> ClearScriptRules:
    """Ленивая загрузка: файл правил на четверть мегабайта."""
    global _RULES
    if _RULES is None:
        with _LOCK:
            if _RULES is None:
                _RULES = ClearScriptRules()
    return _RULES


def translit_to_font(translit: str) -> str:
    return get_rules().translit_to_font(translit)


def available() -> bool:
    return RULES_PATH.exists() and FONT_PATH.exists()


if __name__ == "__main__":
    for w in ["xalimaq", "ulus", "karou", "bolxu", "γazar-yēn", "kelen melen"]:
        print(f"  {w:14s} -> {translit_to_font(w)}")
