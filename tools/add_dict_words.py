# -*- coding: utf-8 -*-
"""
add_dict_words.py — дописать слова в точный словарь, который лежит внутри
model/translit_model.npz, не трогая веса и не переобучая модель.

Словарь проверяется раньше модели (core/transliterate.py), поэтому
добавленное слово сразу начинает переводиться так, как записано:

    python -m tools.add_dict_words өдртн=ödür-tani
    python -m tools.add_dict_words өдртн=ödür-tani хальмгудт=xalimaγud-tu

Существующую запись скрипт заменяет и печатает, что было раньше.
Остальные массивы в .npz переписываются как есть, байт в байт.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

META_KEY = "meta_json"


def add_words(npz_path: Path, pairs: dict) -> None:
    data = np.load(npz_path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    meta = json.loads(bytes(arrays[META_KEY]).decode("utf-8"))
    dictionary = meta["dictionary"]

    for src, tgt in pairs.items():
        old = dictionary.get(src)
        dictionary[src] = tgt
        if old is None:
            print(f"  + {src} -> {tgt}")
        elif old != tgt:
            print(f"  ~ {src} -> {tgt} (было {old})")
        else:
            print(f"  = {src} -> {tgt} (уже есть)")

    raw = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    arrays[META_KEY] = np.frombuffer(raw, dtype=np.uint8)
    np.savez_compressed(npz_path, **arrays)
    print(f"Записан {npz_path}: слов в словаре {len(dictionary)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="кириллица=транслитерация")
    ap.add_argument("--npz", type=Path, default=Path("model/translit_model.npz"))
    args = ap.parse_args()

    pairs = {}
    for p in args.pairs:
        if "=" not in p:
            sys.exit(f"Ожидалось слово=транслитерация, получено: {p}")
        src, tgt = p.split("=", 1)
        # ключи словаря — строчные в NFC, так их ищет core/transliterate.py
        src = unicodedata.normalize("NFC", src.strip().lower())
        tgt = unicodedata.normalize("NFC", tgt.strip())
        if not src or not tgt:
            sys.exit(f"Пустая часть в паре: {p}")
        pairs[src] = tgt
    add_words(args.npz, pairs)


if __name__ == "__main__":
    main()
