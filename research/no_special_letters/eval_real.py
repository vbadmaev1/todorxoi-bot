# -*- coding: utf-8 -*-
"""Реальные примеры: ловит ли правило и что даёт восстановление."""
import csv
src = open('eval2.py').read().split("rnd = random.Random(3)")[0]
g = {}
exec(compile(src, 'eval2_head', 'exec'), g)
text_flags, is_degraded = g['text_flags'], g['is_degraded']
body = open('eval2.py').read()
restore_src = body[body.index('def restore_text'):body.index("print('\\n== Сквозная")]
exec(restore_src, g)
restore_text = g['restore_text']
from eval4 import is_russian  # noqa  (eval4 печатает свои замеры — нам не мешает)

ok = 0; rows = list(csv.DictReader(open('real_examples.tsv'), delimiter='\t'))
print('\n== Реальные примеры')
for r in rows:
    f = text_flags(r['as_written']); k = is_degraded(f)
    rest = restore_text(r['as_written'])
    good = rest == r['correct']; ok += good
    print(f"  {'✓' if good else '✗'} [{k or '—':6s}] {r['as_written']:38s} -> {rest:38s} (надо: {r['correct']})")
print(f'  восстановлено целиком: {ok}/{len(rows)}')
