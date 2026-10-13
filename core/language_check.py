# -*- coding: utf-8 -*-
"""
language_check.py — не русский ли это текст.

Бот и сайт читают кириллицу как калмыцкую. Многие присылают русские фразы
(«Как дела?») или свои имя и фамилию и ждут перевод: получается набор букв,
похожий на калмыцкий, но не слова. Здесь такой ввод узнаётся, чтобы
предупредить человека: сначала переведите текст на калмыцкий.

Три ответа (LANG_*):
  russian — текст на русском; предупреждаем «переведите на калмыцкий»;
  name    — похоже на русское имя или фамилию; предупреждаем мягче:
            запись пойдёт по звучанию калмыцкого слова, результат нужно
            проверить;
  kalmyk  — калмыцкий или непонятно. Тут ничего не говорим: лучше
            промолчать, чем сказать калмыку, что он пишет по-русски.

Решение по словам:
  * буквы ә ө ү һ җ ң — калмыцкий, сразу;
  * слово из словаря калмыцких слов бота — калмыцкое;
  * слово из словаря русских слов (русская сторона корпуса сайта) — русское;
  * имя из словаря pymorphy3 или фамилия по окончанию (-ов, -ева, -ский) —
    имя;
  * остальные — по символьным n-граммам: во сколько раз такие сочетания
    букв чаще в русских словах, чем в калмыцких (в том числе набранных без
    спецбукв).
Данные — model/lang_check.tsv.gz, собирается tools/build_lang_model.py.
Замеры — в docs/language_check.md.
"""

import gzip
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

LANG_RUSSIAN = "russian"
LANG_NAME = "name"
LANG_KALMYK = "kalmyk"

_HERE = Path(__file__).resolve().parent
DEFAULT_DATA = _HERE.parent / "model" / "lang_check.tsv.gz"

_SPECIAL = set("әөүһҗң")
_WORD = re.compile(r"[а-яёәөүһҗң]+(?:-[а-яёәөүһҗң]+)*", re.I)
_ORDERS = (2, 3, 4)
# окончания русских фамилий и отчеств; «Босхомджиев» тоже сюда попадёт —
# калмыцкая фамилия в русской записи, о ней сообщение тоже верное
_SURNAME = re.compile(
    r"(ов|ова|ев|ева|ёв|ёва|ин|ина|ын|ына|ский|ская|цкий|цкая|ович|евич|"
    r"овна|евна|ична|ичев|ых|их|енко|ук|юк)$"
)
# Порог n-граммного счёта (средний log-лог. отношение). По отложенным словам
# при +1.0 русские узнаются в 88%, калмыцкие по ошибке в 1.3%.
NGRAM_RU = 1.0
NGRAM_KAL = -0.3


@dataclass
class LangCheck:
    kind: str = LANG_KALMYK
    # доля слов, давших такой вердикт, — для логов и подбора порогов
    share: float = 0.0
    reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def warn(self) -> bool:
        return self.kind in (LANG_RUSSIAN, LANG_NAME)


class _Data:
    def __init__(self, path):
        self.words: Set[str] = set()
        self.names: Set[str] = set()
        self.grams: Dict[str, float] = {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                kind, key, val = line.rstrip("\n").split("\t")
                if kind == "w":
                    self.words.add(key)
                elif kind == "n":
                    self.names.add(key)
                else:
                    self.grams[key] = float(val)

    def ngram_score(self, word: str) -> float:
        s = f"^{word}$"
        vals = [
            self.grams[s[i : i + n]]
            for n in _ORDERS
            for i in range(len(s) - n + 1)
            if s[i : i + n] in self.grams
        ]
        return sum(vals) / len(vals) if vals else 0.0


_data: Optional[_Data] = None
_data_failed = False


def _load() -> Optional[_Data]:
    global _data, _data_failed
    if _data is None and not _data_failed:
        path = os.environ.get("LANG_CHECK_PATH") or DEFAULT_DATA
        try:
            _data = _Data(path)
        except OSError:
            # без файла проверка молчит: бот работает как раньше
            _data_failed = True
    return _data


def check(text: str) -> LangCheck:
    """Оценивает кириллический текст. Остальные письменности не трогает."""
    res = LangCheck()
    data = _load()
    if data is None:
        return res
    from . import fix_letters

    raw = unicodedata.normalize("NFC", text)
    norm = fix_letters.normalize_lookalikes(raw)
    if _SPECIAL & set(norm.lower()):
        res.reasons["спецбуквы"] = 1
        return res
    # цифры и знаки не слова; одиночные буквы не судим
    toks: List[str] = [t.lower() for t in _WORD.findall(norm) if len(t) >= 2]
    if not toks:
        return res
    fx = fix_letters.get_fixer()
    capitalized = {m.lower() for m in re.findall(r"\b[А-ЯЁ][а-яё]+", norm)}

    ru = name = kal = 0
    for t in toks:
        parts = t.split("-")
        t0 = parts[0] if len(parts) == 1 else t
        # «дж» — русская запись калмыцкой «җ» (Дорджи, Джангар): в калмыцкой
        # орфографии такого сочетания нет, это уже русская запись, пусть и
        # калмыцкого слова. Словарь калмыцких слов тут не аргумент: в нём
        # Дорджи есть, потому что встречается в текстах.
        russian_spelling = "дж" in t0 and len(t0) >= 4
        in_kal = not russian_spelling and (t0 in fx.freq or all(p in fx.freq for p in parts))
        score = data.ngram_score(t0)
        # В русском корпусе попадаются калмыцкие имена и названия
        # (Джангар, Хонгор): если буквосочетания у слова калмыцкие, словарю
        # русских слов не верим.
        in_ru = t0 in data.words and score > NGRAM_KAL
        is_name = t0 in data.names and (t0 in capitalized or not in_ru)
        if in_kal and not (in_ru and not is_name):
            kal += 1
        elif is_name or russian_spelling:
            name += 1
        elif in_ru:
            ru += 1
        elif t0 in capitalized and _SURNAME.search(t0) and len(t0) >= 5:
            name += 1
        elif score >= NGRAM_RU:
            # не из словарей, но по буквам русское: слово это или имя,
            # не знаем — предупреждаем мягко
            name += 1
        elif score <= NGRAM_KAL:
            kal += 1
    n = len(toks)
    res.reasons = {"русские": ru, "имена": name, "калмыцкие": kal, "слов": n}

    if kal / n >= 0.5 or kal >= 1 and kal * 3 > ru + name:
        return res
    if ru >= 1 and (ru + name) / n >= 0.6 and ru >= name:
        res.kind, res.share = LANG_RUSSIAN, (ru + name) / n
    elif name >= 1 and (ru + name) / n >= 0.6:
        res.kind, res.share = LANG_NAME, (ru + name) / n
    return res
