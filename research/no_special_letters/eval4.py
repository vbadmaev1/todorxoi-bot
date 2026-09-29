# -*- coding: utf-8 -*-
"""Фильтр «это русский текст»: служебные слова + доля слов на гласную/й."""
import collections, random, re
src = open('eval2.py').read().split("rnd = random.Random(3)")[0]
g = {}
exec(compile(src, 'eval2_head', 'exec'), g)
held, WORD, text_flags, is_degraded = g['held'], g['WORD'], g['text_flags'], g['is_degraded']
from degrade import degrade_text, normalize_lookalikes, SPECIAL

RU_FUNC = set('и в не на что с по как это для от к о из у за до но же бы ли то так его она он они мы вы ты я '
              'был была было были есть уже ещё еще или если чтобы когда только при над под без через все всё '
              'этот эта эти тот та те мне меня нас вас их ему ей им который которая которые'.split())


def ru_score(text):
    toks = WORD.findall(normalize_lookalikes(text).lower())
    if not toks:
        return 0.0, 0.0
    func = sum(t in RU_FUNC for t in toks) / len(toks)
    long = [t for t in toks if len(t) >= 3]
    vend = sum(t[-1] in 'аеёиоуыэюяй' for t in long) / max(1, len(long))
    return func, vend


def is_russian(text):
    func, vend = ru_score(text)
    return func >= 0.12 or vend >= 0.55


txt = open('/Users/vbadmaev1/Desktop/todorxoi-bot/README.md').read() + open('/Users/vbadmaev1/Desktop/todorxoi-bot/bot/texts.py').read()
ru = [s.strip() for s in re.split(r'[.!?\n]+', txt) if len(re.findall('[а-яё]+', s.lower())) >= 3]
rnd = random.Random(5)
kal = [t for t, nat, _ in rnd.sample(held, 3000) if len(nat) >= 3]
sets = {'русский (README+texts.py)': ru, 'калмыцкий чистый': kal,
        'калмыцкий soft_h': [degrade_text(t, 'soft_h', rnd) for t in kal],
        'калмыцкий bare': [degrade_text(t, 'bare', rnd) for t in kal]}
for name, ss in sets.items():
    f = [ru_score(s) for s in ss]
    r = sum(is_russian(s) for s in ss) / len(ss)
    fm = sum(a for a, _ in f) / len(f); vm = sum(b for _, b in f) / len(f)
    print(f'{name:28s} «русский»: {100*r:5.1f}%   служебных слов {100*fm:4.1f}%   слов на гласную {100*vm:4.1f}%')
print('\nПравило с фильтром (русский → не трогаем):')
for name, ss in sets.items():
    k = sum(bool(is_degraded(text_flags(s))) and not is_russian(s) for s in ss) / len(ss)
    print(f'  {name:28s} помечено испорченным: {100*k:.1f}%')
print('\nрусские, которые фильтр пропустил:', [s[:70] for s in ru if not is_russian(s)][:10])
