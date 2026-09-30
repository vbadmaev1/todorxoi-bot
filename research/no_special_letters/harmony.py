# -*- coding: utf-8 -*-
"""Гармония гласных и парные согласные: г/һ, к/х и другие.

Гипотеза: г пишется только в словах с передними (мягкими) гласными, һ —
только с задними (твёрдыми). Проверяем на проверенном словаре модели и на
частотном словаре корпуса: для каждой согласной — в словах какого ряда она
встречается.

Ряд слова по гласным:
  задний   — есть а, о, у (я, ю, ё в начале = йа, йу, йо) и нет ә ө ү е э;
  передний — есть ә, ө, ү, е, э и нет а, о, у;
  нейтральный — только и;
  смешанный — есть и те, и другие (заимствования, составные слова).
"""
import collections, json, re, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.fix_letters import load_vocab, DEFAULT_VOCAB  # noqa: E402
from tools.build_letters_vocab import read_dictionary  # noqa: E402

BACK, FRONT = set('аоуы'), set('әөүеэ')


def row(w):
    body = w
    # я/ю/ё в начале слова — йотированные задние гласные
    if body[:1] in 'яюё':
        body = {'я': 'а', 'ю': 'у', 'ё': 'о'}[body[0]] + body[1:]
    # после согласной я/ю — тоже задние (буудя, соляд)
    body = body.replace('я', 'а').replace('ю', 'у').replace('ё', 'о')
    b, f = bool(BACK & set(body)), bool(FRONT & set(body))
    if b and f:
        return 'смеш'
    if b:
        return 'задн'
    if f:
        return 'пер'
    return 'нейтр'


vocab, protected, _ = load_vocab(DEFAULT_VOCAB)
trusted = read_dictionary(ROOT / 'model/translit_model.npz')
sources = {'словарь модели (проверенный)': {w: 1 for w in trusted if '-' not in w},
           'корпус (по употреблениям)': {w: n for w, n in vocab.items() if '-' not in w}}

CONS = 'бвгһджҗзйклмнңпрстфхцчшщ'
for name, src in sources.items():
    print(f'\n===== {name}: {len(src)} слов')
    tab = {c: collections.Counter() for c in CONS}
    ex = {c: collections.defaultdict(collections.Counter) for c in CONS}
    for w, n in src.items():
        r = row(w)
        for c in set(w) & set(CONS):
            tab[c][r] += n
            ex[c][r][w] += n
    print(f'  {"":3s} {"задн":>7s} {"пер":>7s} {"нейтр":>7s} {"смеш":>7s}   доля задних среди задн+пер')
    for c in CONS:
        t = tab[c]; bf = t['задн'] + t['пер']
        if not bf:
            continue
        print(f'  {c:3s} {t["задн"]:7d} {t["пер"]:7d} {t["нейтр"]:7d} {t["смеш"]:7d}   {100*t["задн"]/bf:5.1f}%')
    for c in 'гһкхңн':
        for r in ('задн', 'пер'):
            print(f'    {c} в {r}: {[w for w, _ in ex[c][r].most_common(12)]}')

# г и һ: а если смотреть на соседнюю гласную, а не на всё слово?
print('\n===== г/һ/к/х по соседней гласной (корпус, по употреблениям)')
nb = {c: collections.Counter() for c in 'гһкх'}
nbex = {c: collections.defaultdict(collections.Counter) for c in 'гһкх'}
V = 'аоуыяюёәөүеэи'
for w, n in vocab.items():
    for i, c in enumerate(w):
        if c not in nb:
            continue
        # ближайшая гласная справа, иначе слева
        right = next((x for x in w[i + 1:] if x in V), None)
        left = next((x for x in reversed(w[:i]) if x in V), None)
        v = right or left or '-'
        v = {'я': 'а', 'ю': 'у', 'ё': 'о'}.get(v, v)
        nb[c][v] += n
        nbex[c][v][w] += n
for c, t in nb.items():
    tot = sum(t.values())
    print(f'  {c}: ' + '  '.join(f'{v}:{100*k/tot:.1f}%' for v, k in t.most_common()))
for c in 'гһ':
    for v in ('а', 'о', 'у', 'ә', 'ө', 'ү', 'е', 'и'):
        print(f'    {c}+{v}: {[w for w, _ in nbex[c][v].most_common(10)]}')
