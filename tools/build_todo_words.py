# -*- coding: utf-8 -*-
"""
build_todo_words.py — частоты слов тодо бичиг (в транслитерации) для
core/broken_words.py.

Берёт данные модели кириллицы (todorxoi-inference/translit2cyr,
translit2cyr_data.jsonl.gz: словарные пары и предложения корпуса в
транслитерации) и пишет model/todo_words.tsv.gz:

    слово<TAB>частота

Слово тодо бичиг — кусок транслитерации между пробелами и дефисами:
kele-bēr — два слова (kele, bēr), как они и пишутся на странице.

    python -m tools.build_todo_words
    python -m tools.build_todo_words --data путь/translit2cyr_data.jsonl.gz
"""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_DATA = Path.home() / "Desktop/todorxoi-inference/translit2cyr/translit2cyr_data.jsonl.gz"
DEFAULT_OUT = Path("model/todo_words.tsv.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    freq = Counter()
    opener = gzip.open if args.data.suffix == ".gz" else open
    with opener(args.data, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = [(r["cyr"], r["tr"])] if r["kind"] == "pair" else [t for t in r["toks"] if t]
            for cyr, tr in toks:
                if cyr[:1].isalpha():
                    freq.update(p for p in re.split(r"[- ]+", tr) if p)

    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for word, n in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{word}\t{n}\n")
    print(f"{len(freq)} слов -> {args.out} ({args.out.stat().st_size / 1e3:.0f} КБ)")


if __name__ == "__main__":
    main()
