# -*- coding: utf-8 -*-
"""Правильные тексты и слова-двойники (цаган/цаһан, дакад/дәкәд, ...).

1. Что конкретно меняется в правильных предложениях при разных политиках.
2. Пары пользователя: частоты, в каких текстах встречаются, транслитерация.
3. Все коллизии: дают ли два варианта один и тот же тодо бичиг.
"""
import collections, csv, random, re, sys
src = open('eval2.py').read().split("rnd = random.Random(3)")[0]
g = {}
exec(compile(src, 'eval2_head', 'exec'), g)
held, clean, R, WORD, noise, train = g['held'], g['clean'], g['R'], g['WORD'], g['noise'], g['train']
text_flags, is_degraded = g['text_flags'], g['is_degraded']
from degrade import SPECIAL, degrade_text, normalize_lookalikes
sys.path.insert(0, '/Users/vbadmaev1/Desktop/todorxoi-bot')
from core.transliterate import transliterate_word
from core.translit_todo import translit_to_todo
import eval4
is_russian = eval4.is_russian

TOK = re.compile(r"[А-Яа-яЁёӘәӨөҮүҺһҖҗҢң]+(?:-[А-Яа-яЁёӘәӨөҮүҺһҖҗҢң]+)*")


def fix(text, policy):
    """policy: 'oov' — трогаем только незнакомые слова; 'all' — и знакомые
    слова-двойники заменяем на более частый вариант."""
    norm = normalize_lookalikes(text)
    if is_russian(norm) or not is_degraded(text_flags(norm)):
        return norm
    def rep(m):
        w = m.group(0); low = w.lower()
        if policy == 'oov' and low in clean:
            return w
        c = R.candidates(low)
        if not c or c[0] == low:
            return w
        return c[0][:1].upper() + c[0][1:] if w[:1].isupper() else c[0]
    return TOK.sub(rep, norm)


rnd = random.Random(7)
sample = rnd.sample(held, 5000)
print('\n== 1. Правильные предложения (5000 из отложенных книг)')
for policy in ['oov', 'all']:
    ch = collections.Counter(); sents = 0; toks = 0
    for t, nat, _ in sample:
        f = fix(t, policy)
        a, b = WORD.findall(t.lower()), WORD.findall(f.lower())
        toks += len(a)
        if a != b:
            sents += 1
            for x, y in zip(a, b):
                if x != y:
                    ch[(x, y)] += 1
    n_noise = sum(n for (x, y), n in ch.items() if x in noise or (x not in clean and x not in train))
    print(f'  политика {policy}: изменено предложений {100*sents/len(sample):.2f}%, слов {sum(ch.values())} '
          f'({100*sum(ch.values())/toks:.3f}%); из них исходное слово — OCR-шум/незнакомое: {n_noise}')
    print('    ', [f'{x}→{y}' for (x, y), _ in ch.most_common(30)])

# качество на испорченных при тех же политиках
print('  для сравнения, испорченный текст (soft_h / bare), точность по словам:')
for st in ['soft_h', 'bare']:
    for policy in ['oov', 'all']:
        ok = tot = 0
        for t, nat, _ in sample[:1500]:
            d = degrade_text(t, st, rnd); f = fix(d, policy)
            a, b = WORD.findall(t.lower()), WORD.findall(f.lower())
            if len(a) == len(b):
                tot += len(a); ok += sum(x == y for x, y in zip(a, b))
        print(f'    {st:7s} {policy}: {100*ok/tot:.1f}%')

# ---------------------------------------------------------------- пары
print('\n== 2. Пары из вопроса')
PAIRS = [('цаһан', 'цаган'), ('дәкәд', 'дакад'), ('бәрсн', 'барсн'), ('үстә', 'уста'), ('авһ', 'авг')]
# где встречается слово без спецбукв: в тексте, который в остальном написан со спецбуквами?
allsent = []
for line in open(g['CORP']):
    r = g['json'].loads(line)
    allsent.append(r['text'])
for a, b in PAIRS:
    ctx = {a: [], b: []}; cnt = collections.Counter()
    for s in allsent:
        ws = WORD.findall(s.lower())
        for w in (a, b):
            if w in ws:
                cnt[w] += 1
                if len(ctx[w]) < 3:
                    ctx[w].append(s[:110])
    ta, tb = transliterate_word(a), transliterate_word(b)
    same = translit_to_todo(ta) == translit_to_todo(tb)
    print(f'  {a} ({cnt[a]} предл., {ta}) | {b} ({cnt[b]} предл., {tb})  тодо бичиг {"ОДИНАКОВЫЙ" if same else "разный"}; '
          f'{b} в словаре: {b in g["dic"]}, в шуме: {b in noise}')
    for s in ctx[b]:
        print('       ', b, '::', s)

# ---------------------------------------------------------------- все коллизии
print('\n== 3. Все пары-двойники из collisions.tsv: одинаковый ли тодо бичиг')
rows = list(csv.DictReader(open('collisions.tsv'), delimiter='\t'))
same = diff = 0; w_same = w_diff = 0; ex_s = []; ex_d = []
for r in rows:
    a, b = r['correct'], r['degraded_is_also_word']
    ta, tb = transliterate_word(a), transliterate_word(b)
    s = translit_to_todo(ta) == translit_to_todo(tb)
    fa = int(r['freq']) + int(r['freq_other'])
    if s:
        same += 1; w_same += fa
        if len(ex_s) < 25: ex_s.append(f'{a}/{b}={ta}')
    else:
        diff += 1; w_diff += fa
        if len(ex_d) < 25: ex_d.append(f'{a}={ta} / {b}={tb}')
print(f'  пар {len(rows)}: одинаковый тодо бичиг {same} ({100*w_same/(w_same+w_diff):.0f}% употреблений), разный {diff}')
print('  одинаковые:', ex_s)
print('  разные:', ex_d)
