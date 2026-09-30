# -*- coding: utf-8 -*-
"""
broken_words.py — слово, которое распознавание разорвало на два.

В рукописном и старопечатном тексте перо отрывали посреди слова: между
ǰi и kü в örgöǰikü остаётся крошечный просвет. Модель распознавания
училась на шрифтах, где такой просвет — узкий пробел перед суффиксом, и
ставит пробел: «örgöǰi kü», «ged eq», «ye ke», «d ü».

По картинке это не решить. У отрыва пера нет ни одной пустой полоски
поперёк столбца (штрихи перекрываются), но в шрифтах так же выглядят и
настоящие узкие пробелы (972 из 2808 на синтетике), а в шрифте zakaa —
даже 4 % обычных пробелов: хвосты букв заходят под соседнее слово.

Решает язык. Куски a и b склеиваются, если слитное ab встречается в
корпусе не реже, чем более редкий из кусков:

    ged (5)    + eq (0)     -> gedeq (2881)     склеить
    d (0)      + ü (4)      -> dü (9057)        склеить
    örgöǰi (173) + kü (2)   -> örgöǰikü (2)     склеить
    erdeni (183) + dü (9057) -> erdenidü (0)    оставить

На отложенных предложениях корпуса правило склеило бы 29 швов из 30 811
(0.09 %), и почти все такие пары — куски одного кириллического слова
(amita ni, nöl ügei): на кириллицу это не влияет. Незнакомое слитное
слово (старая орфография) ничего не склеивает — остаётся как было.

Частоты — model/todo_words.tsv.gz, собирает tools/build_todo_words.py.
"""

import gzip
import os
import re
import threading
from pathlib import Path
from typing import Dict, Optional

from .translit_todo import todo_to_translit

_HERE = Path(__file__).resolve().parent
_DEFAULT_WORDS = _HERE.parent / "model" / "todo_words.tsv.gz"

_GAP = re.compile("([  ]+)")

_FREQ: Optional[Dict[str, int]] = None
_LOCK = threading.Lock()


def words_path() -> str:
    return os.environ.get("TODO_WORDS_PATH") or str(_DEFAULT_WORDS)


def get_freq() -> Dict[str, int]:
    global _FREQ
    if _FREQ is None:
        with _LOCK:
            if _FREQ is None:
                freq = {}
                with gzip.open(words_path(), "rt", encoding="utf-8") as f:
                    for line in f:
                        word, n = line.rstrip("\n").split("\t")
                        freq[word] = int(n)
                _FREQ = freq
    return _FREQ


# буквы у шва: хвост левого куска и начало правого (знаки препинания вокруг
# не мешают: «örgöǰi kü᠂» — тоже разрыв)
_TAIL = re.compile("[\u1820-\u18aa]+$")
_HEAD = re.compile("^[\u1820-\u18aa]+")


def should_join(a: str, b: str, ab: str, freq: Dict[str, int]) -> bool:
    """a, b — транслитерация кусков у шва, ab — слитного слова (считается
    отдельно: x/k и g/γ зависят от соседней буквы)."""
    n = freq.get(ab, 0)
    return n > 0 and n >= min(freq.get(a, 0), freq.get(b, 0))


def join_broken_words(todo: str) -> str:
    """Строка тодо бичиг (столбец) -> та же строка без разрывов внутри слов.

    Решение принимается по транслитерации, но убирается сам промежуток в
    тодо бичиг: и транслитерация, и кириллица дальше строятся уже из
    исправленной строки."""
    parts = _GAP.split(todo)
    if len(parts) < 3:
        return todo
    freq = get_freq()
    out = [parts[0]]
    for gap, word in zip(parts[1::2], parts[2::2]):
        tail, head = _TAIL.search(out[-1]), _HEAD.match(word)
        if tail and head and should_join(
            todo_to_translit(tail.group()), todo_to_translit(head.group()),
            todo_to_translit(tail.group() + head.group()), freq,
        ):
            out[-1] += word
        else:
            out += [gap, word]
    return "".join(out)


def warmup() -> int:
    return len(get_freq())
