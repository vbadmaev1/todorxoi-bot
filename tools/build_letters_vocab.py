# -*- coding: utf-8 -*-
"""
build_letters_vocab.py — словарь с частотами для core/fix_letters.py.

Берёт предложения корпуса (todorxoi-inference/corpus_out) и словарь модели
транслитерации, пишет model/letters_vocab.tsv.gz:

    слово<TAB>частота<TAB>d|k|r

d — слово из проверенного словаря модели, k — калмыцкое слово из корпуса,
r — русское слово из калмыцких текстов (гражданск, доктор, Сталинград): его
исправлять нельзя, хотя выглядит оно как калмыцкое без спецбукв.

Корпус распознан OCR и сам местами без спецбукв: «гиж» в нём 281 раз,
«болж» 241, «кун», «менгн», латинская h («темдгтэhер»). Такие формы в
словарь попасть не должны, иначе восстановитель будет считать их
правильными. Отсеиваются:
  * слова с латиницей и двойниками — целиком, вместе с обрывками вокруг;
  * слова с «ё», «яя», «юю», «э» не в начале — в калмыцкой орфографии
    их не бывает;
  * слова, у которых есть вариант со спецбуквами как минимум в 5 раз
    частотнее (гиж -> гиҗ). Словарь модели проверен вручную — его слова не
    отсеиваются никогда.

    python -m tools.build_letters_vocab
    python -m tools.build_letters_vocab --corpus путь/corpus_sentences.jsonl
"""

import argparse
import gzip
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from core.fix_letters import SPECIAL, LetterFixer, _ANY_TOKEN, _FOREIGN

DEFAULT_CORPUS = Path.home() / "Desktop/todorxoi-inference/corpus_out/corpus_sentences.jsonl"
DEFAULT_NPZ = Path("model/translit_model.npz")
DEFAULT_OUT = Path("model/letters_vocab.tsv.gz")

_KALMYK = re.compile(r"^[а-яёәөүһҗң]+(?:-[а-яёәөүһҗң]+)*$")
# «э» не в начале слова в калмыцкой орфографии не бывает (после согласной
# пишут «е»); в корпусе это OCR: һаэр (һазр), колхоэин, бээсн
_NEVER = re.compile("ё|яя|юю|(?<=[^-э])э")
NOISE_RATIO = 5


def read_corpus(path, exclude_books=()):
    """-> (частоты калмыцких слов, частоты русских слов). Книги из
    exclude_books пропускаются — так строится словарь для честной проверки."""
    kal, ru = Counter(), Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["ocr_suspect"] or r["book_id"] in exclude_books:
                continue
            foreign = {unicodedata.normalize("NFC", f["word"].lower())
                       for f in r["foreign_words"] if f["kind"] in ("ru_word", "ru_name")}
            for tok in _ANY_TOKEN.findall(unicodedata.normalize("NFC", r["text"])):
                if _FOREIGN.search(tok):
                    continue  # «темдгтэhер»: и само слово, и его куски — мусор
                w = tok.lower().strip("-")
                if not _KALMYK.match(w):
                    continue
                (ru if w in foreign else kal)[w] += 1
    return kal, ru


def read_dictionary(npz_path):
    meta = json.loads(bytes(np.load(npz_path)["meta_json"]).decode("utf-8"))
    return {unicodedata.normalize("NFC", k) for k in meta["dictionary"]}


def build_vocab(corpus_path, npz_path, exclude_books=()):
    """-> (vocab: слово -> частота, protected: русские слова,
    noise: отсеянные формы, trusted: слова проверенного словаря)."""
    kal, ru = read_corpus(corpus_path, exclude_books)
    trusted = read_dictionary(npz_path)
    vocab = Counter({w: n for w, n in kal.items() if w in trusted or not _NEVER.search(w)})
    for w in trusted:
        vocab[w] = max(vocab[w], 1)

    fixer = LetterFixer(vocab)
    noise = set()
    for w, n in vocab.items():
        if w in trusted:
            continue
        for v, _, subs in fixer.candidates(w):
            if v != w and SPECIAL & set(v) and vocab[v] >= NOISE_RATIO * n:
                noise.add(w)
                break
    vocab = {w: n for w, n in vocab.items() if w not in noise}
    protected = {w for w in ru if w not in vocab}
    return vocab, protected, noise, trusted


def write_vocab(path, vocab, protected, trusted):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for w, n in sorted(vocab.items(), key=lambda x: (-x[1], x[0])):
            fh.write(f"{w}\t{n}\t{'d' if w in trusted else 'k'}\n")
        for w in sorted(protected):
            fh.write(f"{w}\t0\tr\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    vocab, protected, noise, trusted = build_vocab(args.corpus, args.npz)
    write_vocab(args.out, vocab, protected, trusted)
    print(f"слов {len(vocab)}, русских {len(protected)}, отсеяно как шум {len(noise)}")
    print(f"записан {args.out} ({args.out.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
