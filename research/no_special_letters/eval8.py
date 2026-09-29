# -*- coding: utf-8 -*-
"""Повтор замеров на другой выборке (книги задаются через HELD) и выгрузка
случаев, в которых восстановитель не уверен: uncertain.tsv.

    HELD=книга1,книга2 SEED=42 python eval8.py
"""
import collections, csv, os, random
src = open('eval7.py').read().split("rnd = random.Random(7)")[0]
g = {}
exec(compile(src, 'eval7_head', 'exec'), g)
held, clean, R, WORD, TOK, fix, subs = g['held'], g['clean'], g['R'], g['WORD'], g['TOK'], g['fix'], g['subs']
text_flags, is_degraded, is_russian = g['text_flags'], g['is_degraded'], g['is_russian']
from degrade import PARTIAL, SPECIAL, degrade_text, normalize_lookalikes

SEED = int(os.environ.get('SEED', '42'))
rnd = random.Random(SEED)
sample = rnd.sample(held, min(4000, len(held)))
print(f'\nотложено: {os.environ.get("HELD", "по умолчанию")}; предложений {len(held)}, в выборке {len(sample)}, seed {SEED}')

# ---------------------------------------------------------------- правильные
print('\n== Правильные предложения')
flagged = 0; changed = collections.Counter(); toks = 0
for t, nat, _ in sample:
    flagged += bool(is_degraded(text_flags(t))) and not is_russian(t)
    f = fix(t, 'habits')
    a, b = WORD.findall(t.lower()), WORD.findall(f.lower())
    toks += len(a)
    for x, y in zip(a, b):
        if x != y:
            changed[(x, y)] += 1
real = {k: n for k, n in changed.items() if k[0] in clean}
print(f'  помечено испорченными: {100*flagged/len(sample):.2f}%; изменено слов {sum(changed.values())} '
      f'({100*sum(changed.values())/toks:.3f}%), из них словарных (настоящих) слов: {sum(real.values())}')
print('  ', [f'{x}→{y}' for (x, y), _ in changed.most_common(25)])
if real:
    print('   словарные:', [f'{x}→{y}' for (x, y), _ in collections.Counter(real).most_common(20)])

# ---------------------------------------------------------------- обнаружение
print('\n== Обнаружение испорченного текста')
for st in ['soft_h', 'soft_g', 'bare', 'e_style', 'russian', 'mixed'] + ['partial:' + p for p in PARTIAL]:
    r2 = random.Random(SEED); hit = n = 0
    for t, nat, _ in sample:
        if not SPECIAL & set(''.join(nat)):
            continue
        keep = frozenset(SPECIAL - PARTIAL[st[8:]]) if st.startswith('partial:') else frozenset()
        d = degrade_text(t, 'soft_h' if keep else st, r2, keep)
        if d == t:
            continue
        n += 1; hit += bool(is_degraded(text_flags(d))) and not is_russian(d)
    print(f'  {st:18s} поймано {100*hit/n:5.1f}%')

# ---------------------------------------------------------------- точность + сомнительные
print('\n== Точность по словам (oov / all / habits) и сомнительные случаи')
rows = []
for st in ['soft_h', 'soft_g', 'bare', 'e_style', 'mixed']:
    res = {}
    for policy in ['oov', 'all', 'habits']:
        r2 = random.Random(SEED + 1); ok = tot = 0
        for t, nat, _ in sample[:1500]:
            d = degrade_text(t, st, r2); f = fix(d, policy)
            a, dd, b = WORD.findall(t.lower()), WORD.findall(normalize_lookalikes(d).lower()), WORD.findall(f.lower())
            if not (len(a) == len(b) == len(dd)):
                continue
            tot += len(a); ok += sum(x == y for x, y in zip(a, b))
            if policy != 'habits' or not (is_degraded(text_flags(d)) and not is_russian(d)):
                continue
            for x, y0, y in zip(a, dd, b):
                c = R.candidates(y0)
                spec = [v for v in c if v != y0]
                if y0 in clean and spec:
                    cat = 'двойник: запись сама — слово'
                elif y0 not in clean and len(c) >= 2 and clean[c[1]] * 5 >= clean[c[0]]:
                    cat = 'кандидаты близки по частоте'
                elif y0 not in clean and not c and y0 != x:
                    cat = 'кандидатов нет'
                else:
                    continue
                rows.append(dict(style=st, category=cat, written=y0, truth=x, chosen=y, correct=int(x == y),
                                 candidates=' '.join(f'{v}:{clean[v]}' for v in c[:5]) or '-',
                                 sentence=d))
        res[policy] = 100 * ok / tot
    print(f'  {st:8s} ' + '  '.join(f'{p} {v:.1f}%' for p, v in res.items()))

out = f'uncertain_{SEED}.tsv'
with open(out, 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
print(f'\nсомнительных случаев: {len(rows)} -> {out}')
cc = collections.Counter(r['category'] for r in rows)
for cat, n in cc.most_common():
    ok = sum(r['correct'] for r in rows if r['category'] == cat)
    print(f'  {cat:30s} {n:5d}  верно {100*ok/n:.0f}%')
