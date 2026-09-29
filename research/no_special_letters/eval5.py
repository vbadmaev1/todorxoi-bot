# -*- coding: utf-8 -*-
"""Согласная+я/ю в НЕЗНАКОМОМ слове и «йа/йу/йо» в начале: насколько жёсткие триггеры."""
import collections, random, re
src = open('eval2.py').read().split("rnd = random.Random(3)")[0]
g = {}
exec(compile(src, 'eval2_head', 'exec'), g)
held, clean, CONS = g['held'], g['clean'], g['CONS']
from degrade import degrade, normalize_lookalikes, SPECIAL
C = ''.join(CONS)
toks = [w for _, nat, _ in held for w in nat]
oov = [w for w in toks if w not in clean]
rnd = random.Random(0)
deg = [degrade(w, 'soft_h') for w in toks]
deg_oov = [d for d in deg if d not in clean]
for name, rx in [('C+я', f'[{C}]я'), ('C+ю', f'[{C}]ю'), ('C+я|C+ю', f'[{C}][яю]')]:
    a = sum(bool(re.search(rx, w)) for w in oov)
    b = sum(bool(re.search(rx, w)) for w in deg_oov)
    print(f'{name:8s} в незнакомых словах: чистый текст {a} ({100*a/len(toks):.3f}% всех токенов) | '
          f'испорченный {b} ({100*b/len(toks):.1f}%)  примеры чистых: {collections.Counter(w for w in oov if re.search(rx, w)).most_common(10)}')
allw = collections.Counter(toks)
for p in ['йа', 'йу', 'йо', 'йэ', 'йе']:
    n = sum(c for w, c in clean.items() if w.startswith(p))
    print(f'слова на «{p}» в словаре: {n}  примеры {[w for w, _ in collections.Counter({w: c for w, c in clean.items() if w.startswith(p)}).most_common(8)]}')
