# -*- coding: utf-8 -*-
"""Замеры на корпусе: какие триггеры надёжны, сколько коллизий, как хорошо
восстанавливается. Держим 2 книги в стороне — словарь строится без них."""
import collections
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

from degrade import (CONS, PARTIAL, SPECIAL, STYLES, Restorer, degrade,
                     degrade_text, normalize_lookalikes)

CORP = '/Users/vbadmaev1/Desktop/todorxoi-inference/corpus_out/corpus_sentences.jsonl'
NPZ = '/Users/vbadmaev1/Desktop/todorxoi-bot/model/translit_model.npz'
HELD = {'бадмин_үкрчә_теегин_республик', 'эрнҗана_ботхн_һалан_хадһл'}
WORD = re.compile(r"[а-яёәөүһҗң]+(?:-[а-яёәөүһҗң]+)*")

train = collections.Counter(); ru = collections.Counter()
held = []  # (text, native tokens)
for line in open(CORP):
    r = json.loads(line)
    if r['ocr_suspect']:
        continue
    fw = {unicodedata.normalize('NFC', f['word'].lower()) for f in r['foreign_words']}
    t = unicodedata.normalize('NFC', r['text'])
    toks = WORD.findall(t.lower())
    nat = [w for w in toks if w not in fw]
    for w in toks:
        if w in fw:
            ru[w] += 1
    if r['book_id'] in HELD:
        held.append((t, nat))
    else:
        train.update(nat)
d = np.load(NPZ)
dic = json.loads(bytes(d['meta_json']).decode())['dictionary']
for k in dic:
    k = unicodedata.normalize('NFC', k)
    train[k] += 0  # есть в словаре, но без частоты
for k in list(train):
    if train[k] == 0:
        train[k] = 1
R = Restorer(train)
print(f'train vocab {len(train)}  held-out sentences {len(held)}  ru tokens {sum(ru.values())}')


# ---------------------------------------------------------------- триггеры
def has_front(w):
    return bool(re.search('[әөүэе]', w))


def has_back(w):
    return bool(re.search('[аоуыяёю]', w[1:] if w[:1] in 'яюёе' else w))


TRIG = {
    'C+я': lambda w: bool(re.search(f'[{"".join(CONS)}]я', w)),
    'C+ю': lambda w: bool(re.search(f'[{"".join(CONS)}]ю', w)),
    'C+ё': lambda w: bool(re.search(f'[{"".join(CONS)}]ё', w)),
    'ё где угодно': lambda w: 'ё' in w,
    'яя/юю/ёё': lambda w: bool(re.search('яя|юю|ёё', w)),
    'гармония э/е + а/о/у': lambda w: bool(re.search('[эе]', w)) and bool(re.search('[аоуы]', w)),
    'гармония ә/ө/ү + а/о/у': lambda w: bool(re.search('[әөү]', w)) and bool(re.search('[аоуы]', w)),
    'дж': lambda w: 'дж' in w,
    'нг': lambda w: 'нг' in w,
}


def rate(tokens, f):
    n = sum(1 for w in tokens if f(w))
    return n, 100 * n / max(1, len(tokens))


held_tok = [w for _, nat in held for w in nat]
ru_tok = list(ru.elements())
rnd = random.Random(0)
deg = {}
for st in ['soft_h', 'soft_g', 'bare', 'e_style', 'russian', 'mixed']:
    deg[st] = [(w, normalize_lookalikes(degrade_text(w, st, rnd)).lower()) for w in held_tok]

print('\n== Триггеры (доля токенов, %): чистый калмыцкий | русские вкрапления | испорченный soft_h | bare')
for name, f in TRIG.items():
    c = rate(held_tok, f)[1]; r_ = rate(ru_tok, f)[1]
    s = rate([d for _, d in deg['soft_h']], f)[1]; b = rate([d for _, d in deg['bare']], f)[1]
    print(f'  {name:28s} {c:6.2f} | {r_:6.2f} | {s:6.2f} | {b:6.2f}')

# примеры «ложных» срабатываний на чистом калмыцком
for name in ['C+я', 'C+ю', 'ё где угодно', 'яя/юю/ёё', 'гармония э/е + а/о/у', 'дж', 'нг']:
    ex = collections.Counter(w for w in held_tok if TRIG[name](w))
    print(f'  чистые слова, где срабатывает «{name}»: {[w for w, _ in ex.most_common(15)]}')

# ---------------------------------------------------------------- словарный триггер
print('\n== Словарный триггер: слова нет в словаре, но есть правильный кандидат со спецбуквами')
oov_clean = [w for w in held_tok if w not in train]
fp = collections.Counter(w for w in oov_clean if any(v != w for v in R.candidates(w)))
print(f'  чистый: OOV {100*len(oov_clean)/len(held_tok):.1f}% токенов, из них «восстанавливаемых» {sum(fp.values())} '
      f'({100*sum(fp.values())/len(held_tok):.2f}% всех токенов)')
print('  примеры:', [(w, R.candidates(w)[:2]) for w, _ in fp.most_common(12)])

# ---------------------------------------------------------------- восстановление
print('\n== Восстановление по словарю (топ-1 по частоте), по стилям порчи')
cat_ex = collections.defaultdict(collections.Counter)
for st, pairs in deg.items():
    cats = collections.Counter(); ok = collections.Counter()
    for orig, d in pairs:
        if d == orig:
            cat = 'не изменилось'
        else:
            cands = R.candidates(d)
            if d in train:
                cat = 'коллизия (порча = другое слово)'
            elif not cands:
                cat = 'нет кандидатов'
            elif len(cands) == 1:
                cat = 'однозначно'
            else:
                cat = 'несколько кандидатов'
            if cands and cands[0] == orig:
                ok[cat] += 1
            if st == 'soft_h':
                cat_ex[cat][(d, orig)] += 1
        cats[cat] += 1
        if cat == 'не изменилось':
            ok[cat] += 1
    tot = len(pairs)
    acc = sum(ok.values()) / tot
    print(f'  {st:8s} точность {100*acc:.1f}%  ' + '  '.join(
        f'{c}: {100*n/tot:.1f}% (верно {100*ok[c]/max(1,n):.0f}%)' for c, n in cats.most_common()))
for cat, ex in cat_ex.items():
    if cat != 'не изменилось':
        print(f'  [{cat}]', [f'{d}←{o}' for (d, o), _ in ex.most_common(15)])

# ---------------------------------------------------------------- коллизии
print('\n== Самые частые коллизии: правильное слово -> порча, совпавшая с другим словом')
coll = collections.Counter()
for w, n in train.most_common(20000):
    if not SPECIAL & set(w):
        continue
    for st in ['soft_h', 'soft_g', 'bare', 'e_style']:
        dd = degrade(w, st)
        dd = dd.replace('h', 'һ')
        if dd != w and dd in train and train[dd] >= 3:
            coll[(w, dd)] += n
print('  ', [f'{a}→{b} ({train[b]} раз как самостоятельное слово)' for (a, b), _ in coll.most_common(40)])

# ---------------------------------------------------------------- предложения
print('\n== Доля ЧИСТЫХ предложений без единой спецбуквы, по длине (в словах)')
by_len = collections.defaultdict(lambda: [0, 0])
for t, nat in held:
    n = len(nat)
    b = n if n < 8 else (8 if n < 12 else 12)
    by_len[b][0] += 1
    by_len[b][1] += not (SPECIAL & set(''.join(nat)))
for b in sorted(by_len):
    a, z = by_len[b]
    lab = f'{b}' if b < 8 else ('8-11' if b == 8 else '12+')
    print(f'  {lab:5s} слов: {a:6d} предложений, без спецбукв {100*z/a:5.1f}%')

# ---------------------------------------------------------------- примеры
rnd = random.Random(1)
out = open('examples.tsv', 'w')
out.write('style\tcorrect\tdegraded\n')
# В эталон — только предложения, в которых нет ошибок распознавания: в книгах
# встречаются «менгн» (мөңгн), «темдгтэhер», «hаэр» (һазр), «Манж» (Манҗ).
# Отсеиваем всё, что боевой исправитель помечает или меняет.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.fix_letters import get_fixer
_fx = get_fixer()
def _clean(t):
    r = _fx.fix(t)
    return not r.detection.flag and not r.fixes and not any(w not in _fx.freq for w in WORD.findall(t.lower()))
sample = rnd.sample([t for t, nat in held if 5 <= len(nat) <= 25 and SPECIAL & set(''.join(nat)) and _clean(t)], 400)
styles = list(STYLES)[:-1] if '_tmp' in STYLES else list(STYLES)
styles = [s for s in styles if s != '_tmp']
for i, t in enumerate(sample):
    st = styles[i % len(styles)]
    out.write(f'{st}\t{t}\t{degrade_text(t, st, rnd)}\n')
    if i % 4 == 0:
        pk = list(PARTIAL)[(i // 4) % len(PARTIAL)]
        out.write(f'partial:{pk}\t{t}\t{degrade_text(t, "soft_h", rnd, keep=frozenset(SPECIAL - PARTIAL[pk]))}\n')
out.close()
print('\nexamples.tsv написан')
