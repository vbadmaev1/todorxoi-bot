# -*- coding: utf-8 -*-
"""Двойники в испорченном тексте: учитывать привычки автора.

Незнакомые слова восстанавливаются однозначно и выдают, как этот человек
заменяет спецбуквы (ә->я, һ->h, ...). Знакомое слово-двойник («дакад»)
меняем на вариант со спецбуквами только если все нужные для этого замены
автор уже делал в этом же тексте. Кто пишет «бяядл», у того «а» — это «а».
"""
import collections, random, re
from functools import lru_cache
src = open('eval2.py').read().split("rnd = random.Random(3)")[0]
g = {}
exec(compile(src, 'eval2_head', 'exec'), g)
held, clean, R, WORD = g['held'], g['clean'], g['R'], g['WORD']
text_flags, is_degraded = g['text_flags'], g['is_degraded']
from degrade import REPLACEMENTS, SPECIAL, degrade_text, normalize_lookalikes
import eval4
is_russian = eval4.is_russian
TOK = re.compile(r"[А-Яа-яЁёӘәӨөҮүҺһҖҗҢң]+(?:-[А-Яа-яЁёӘәӨөҮүҺһҖҗҢң]+)*")


def subs(v, w):
    """Какие замены спецбукв превращают v в w: множество (буква, замена)."""
    @lru_cache(None)
    def go(i, j):
        if i == len(v):
            return frozenset() if j == len(w) else None
        ch = v[i]
        for o in (REPLACEMENTS.get(ch, []) + [ch]) if ch in REPLACEMENTS else [ch]:
            if w.startswith(o, j):
                rest = go(i + 1, j + len(o))
                if rest is not None:
                    return rest | ({(ch, o)} if ch in SPECIAL and o != ch else set())
        return None
    return go(0, 0)


def fix(text, policy):
    norm = normalize_lookalikes(text)
    if is_russian(norm) or not is_degraded(text_flags(norm)):
        return norm
    toks = TOK.findall(norm)
    habits = collections.Counter()
    for w in toks:
        low = w.lower()
        if low not in clean:
            c = R.candidates(low)
            if c:
                habits.update(subs(c[0], low) or ())

    def rep(m):
        w = m.group(0); low = w.lower()
        c = R.candidates(low)
        if not c or c[0] == low:
            return w
        best = c[0]
        if low in clean:
            if policy == 'oov':
                return w
            if policy == 'habits':
                best = next((v for v in c if v != low and all(habits[s] > 0 for s in (subs(v, low) or ()))), None)
                if best is None or clean[best] < clean[low]:
                    return w
        return best[:1].upper() + best[1:] if w[:1].isupper() else best
    return TOK.sub(rep, norm)


rnd = random.Random(7)
sample = rnd.sample(held, 3000)
print('\n== Точность по словам на испорченном тексте; в скобках — сколько знакомых слов испорчено нами')
print(f'  {"стиль":9s}' + ''.join(f'{p:>20s}' for p in ['oov', 'all', 'habits']))
for st in ['soft_h', 'soft_g', 'bare', 'e_style', 'mixed']:
    row = []
    for policy in ['oov', 'all', 'habits']:
        r2 = random.Random(11)
        ok = tot = broke = 0
        for t, nat, _ in sample[:1500]:
            d = degrade_text(t, st, r2); f = fix(d, policy)
            a, dd, b = WORD.findall(t.lower()), WORD.findall(normalize_lookalikes(d).lower()), WORD.findall(f.lower())
            if len(a) == len(b) == len(dd):
                tot += len(a); ok += sum(x == y for x, y in zip(a, b))
                # слово пришло без порчи (цаган было цаган), а мы его поменяли
                broke += sum(x == y0 and y != x for x, y0, y in zip(a, dd, b))
        row.append(f'{100*ok/tot:.1f}% ({broke})')
    print(f'  {st:9s}' + ''.join(f'{x:>20s}' for x in row))

print('\n== Примеры: правильные слова без спецбукв внутри испорченного текста')
tests = ['Сул цаган шатр наадҗ давулдмн.', 'Дакад нег умшҗ хәләчкәд, бичсән шуулад хайҗ оркв.',
         'Арһан барсн өвгн иләр болн йосар эс болхла, эврәннь мөрән хулхалҗ авхар шиидв.',
         'Адуч йова йовҗ майкан тәәләд, уста бочкур одв.', 'Ода деерән мана күүкдлә таньлдҗ авг.']
for t in tests:
    for st in ['soft_h', 'bare']:
        d = degrade_text(t, st, random.Random(1))
        print(f'  [{st}] {d}\n      oov:    {fix(d, "oov")}\n      habits: {fix(d, "habits")}')
