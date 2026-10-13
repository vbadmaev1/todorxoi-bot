# -*- coding: utf-8 -*-
"""
build_lang_model.py — данные для core/language_check.py: русский это текст
или калмыцкий.

Берёт русскую сторону параллельных корпусов сайта «Кергтә» (ru ↔ хальмг),
словарь калмыцких слов бота и русские имена из словаря pymorphy3, пишет
model/lang_check.tsv.gz:

    w<TAB>слово<TAB>0           русское слово (≥ 2 раз в русском корпусе и
                                 заметно чаще, чем в калмыцком)
    n<TAB>имя<TAB>0             русское имя, фамилия или отчество
    g<TAB>n-грамма<TAB>вес      log(P(n-грамма | русское) / P(… | калмыцкое)),
                                 n = 2..4, края слова — «^» и «$»

n-граммы нужны для слов, которых нет в списках: «Собака», «Манчестер»,
«Дерево». Калмыцкая сторона обучения — слова словаря бота (с частотой,
сглаженной корнем) и те же слова, испорченные так, как пишут люди без ә ө ү
һ җ ң (research/no_special_letters/degrade.py): иначе «сяян» будет похож на
русское слово.

Запускать окружением, где стоит pymorphy3 (например, окружение сайта):

    python -m tools.build_lang_model \\
        --parallel ~/Desktop/kergtya/backend/data/memory/parallel_memory.jsonl \\
        --examples ~/Desktop/kergtya/backend/data/dictionaries/corpus_examples.jsonl

Отдельно печатает точность на отложенных словах — по ней выбираются пороги.
"""

import argparse
import gzip
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "model" / "lang_check.tsv.gz"
VOCAB = ROOT / "model" / "letters_vocab.tsv.gz"

RU_CYR = "а-яё"
WORD = re.compile(f"[{RU_CYR}]+(?:-[{RU_CYR}]+)*")
ORDERS = (2, 3, 4)
MIN_NGRAM = 8  # n-грамма, виденная реже, веса не получает
SMOOTH = 0.5


def russian_counts(paths):
    cnt = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                text = row.get("ru") or row.get("russian") or ""
                cnt.update(WORD.findall(text.lower()))
    return cnt


def kalmyk_vocab():
    """-> {слово: частота} только правильных слов (d, k)."""
    out = {}
    with gzip.open(VOCAB, "rt", encoding="utf-8") as fh:
        for line in fh:
            w, n, kind = line.rstrip("\n").split("\t")
            if kind != "r":
                out[w] = int(n)
    return out


def russian_names():
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    names = set()
    for word, tag, normal, *_ in morph.dictionary.iter_known_words():
        if word != normal or not word.isalpha():
            continue
        if {"Name", "Surn", "Patr"} & set(tag.grammemes) and len(word) >= 3:
            names.add(word)
    return names


def ngrams(word):
    s = f"^{word}$"
    for n in ORDERS:
        for i in range(len(s) - n + 1):
            yield s[i : i + n]


def degraded_variants(words, rnd):
    from research.no_special_letters import degrade

    out = []
    for w in words:
        if not set("әөүһҗң") & set(w):
            continue
        try:
            d = degrade.degrade(w, rnd.choice(list(degrade.STYLES)), rnd)
            if re.fullmatch(f"[{RU_CYR}әөүһҗң]+", d):
                out.append(d)
        except Exception:
            continue
    return out


def train(ru_words, kal_words):
    """ru_words, kal_words — списки (слово, вес). -> {n-грамма: llr}."""
    cru, ckal = Counter(), Counter()
    for w, wt in ru_words:
        for g in ngrams(w):
            cru[g] += wt
    for w, wt in kal_words:
        for g in ngrams(w):
            ckal[g] += wt
    tru, tkal = sum(cru.values()), sum(ckal.values())
    vocab = set(cru) | set(ckal)
    model = {}
    for g in vocab:
        if cru[g] + ckal[g] < MIN_NGRAM:
            continue
        p_ru = (cru[g] + SMOOTH) / (tru + SMOOTH * len(vocab))
        p_kal = (ckal[g] + SMOOTH) / (tkal + SMOOTH * len(vocab))
        model[g] = math.log(p_ru / p_kal)
    return model


def score(model, word):
    gs = [model[g] for g in ngrams(word) if g in model]
    return sum(gs) / max(1, len(gs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rnd = random.Random(7)

    ru_cnt = russian_counts([args.parallel, args.examples])
    kal = kalmyk_vocab()
    ru_total = sum(ru_cnt.values())
    kal_total = sum(kal.values())
    print(f"русских слов {len(ru_cnt)}, калмыцких {len(kal)}")

    # русский словарь: дважды и чаще, и заметно чаще, чем у калмыков
    lexicon = {}
    for w, n in ru_cnt.items():
        if n < 2 or len(w) < 2:
            continue
        k = kal.get(w, 0)
        if k and (k / kal_total) * 5 > n / ru_total:
            continue
        lexicon[w] = n
    print("в словаре русских слов:", len(lexicon))

    names = russian_names()
    print("имён и фамилий:", len(names))

    # обучающие слова: отложим 10%, чтобы честно померить
    ru_words = [(w, 1.0) for w in ru_cnt if len(w) >= 3 and ru_cnt[w] >= 2]
    kal_words = [(w, math.sqrt(n)) for w, n in kal.items() if len(w) >= 3]
    kal_clean = [w for w, _ in kal_words]
    deg = degraded_variants(rnd.sample(kal_clean, min(40000, len(kal_clean))), rnd)
    kal_words += [(w, 1.0) for w in deg if len(w) >= 3]
    rnd.shuffle(ru_words)
    rnd.shuffle(kal_words)
    cut_r, cut_k = len(ru_words) // 10, len(kal_words) // 10
    test_r, train_r = ru_words[:cut_r], ru_words[cut_r:]
    test_k, train_k = kal_words[:cut_k], kal_words[cut_k:]

    model = train(train_r, train_k)
    for thr in (-0.5, 0.0, 0.5, 1.0):
        tp = sum(score(model, w) > thr for w, _ in test_r) / len(test_r)
        fp = sum(score(model, w) > thr for w, _ in test_k) / len(test_k)
        print(f"порог {thr:+.1f}: русские узнаны {tp:.1%}, калмыцкие приняты за русские {fp:.1%}")

    model = train(ru_words, kal_words)
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for w in sorted(lexicon):
            fh.write(f"w\t{w}\t0\n")
        for w in sorted(names):
            fh.write(f"n\t{w}\t0\n")
        for g, v in sorted(model.items()):
            fh.write(f"g\t{g}\t{v:.2f}\n")
    print("записано:", args.out, Path(args.out).stat().st_size // 1024, "КБ")


if __name__ == "__main__":
    main()
