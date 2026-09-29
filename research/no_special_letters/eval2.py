# -*- coding: utf-8 -*-
"""Второй замер: очищенный словарь, правило «текст испорчен», сквозная точность,
и что делает с испорченным вводом текущая модель."""
import collections, json, random, re, sys, unicodedata
import numpy as np
from degrade import (CONS, PARTIAL, SPECIAL, Restorer, degrade, degrade_text,
                     normalize_lookalikes, LOOKALIKE)

sys.path.insert(0, '/Users/vbadmaev1/Desktop/todorxoi-bot')
CORP = '/Users/vbadmaev1/Desktop/todorxoi-inference/corpus_out/corpus_sentences.jsonl'
NPZ = '/Users/vbadmaev1/Desktop/todorxoi-bot/model/translit_model.npz'
import os
HELD = set(os.environ.get('HELD', 'бадмин_үкрчә_теегин_республик,эрнҗана_ботхн_һалан_хадһл').split(','))
WORD = re.compile(r"[а-яёәөүһҗң]+(?:-[а-яёәөүһҗң]+)*")

train = collections.Counter(); held = []
for line in open(CORP):
    r = json.loads(line)
    if r['ocr_suspect']:
        continue
    fw = {unicodedata.normalize('NFC', f['word'].lower()) for f in r['foreign_words']}
    t = unicodedata.normalize('NFC', r['text'])
    toks = WORD.findall(t.lower())
    nat = [w for w in toks if w not in fw]
    (held.append((t, nat, toks)) if r['book_id'] in HELD else train.update(nat))
dic = {unicodedata.normalize('NFC', k) for k in json.loads(bytes(np.load(NPZ)['meta_json']).decode())['dictionary']}
for k in dic:
    train[k] = max(train[k], 1)

# --- очистка: слово корпуса вне словаря, у которого есть «исправленная» форма
# со спецбуквами в 5+ раз частотнее, — это OCR/опечатка, не слово
R0 = Restorer(train)
noise = set()
for w, n in train.items():
    if w in dic or SPECIAL & set(w) and False:
        continue
    for v in R0.candidates(w):
        if v != w and train[v] >= 5 * n:
            noise.add(w); break
clean = collections.Counter({w: n for w, n in train.items() if w not in noise})
R = Restorer(clean)
print(f'словарь {len(train)} -> после чистки {len(clean)}; выброшено как шум {len(noise)}:',
      sorted(noise, key=lambda w: -train[w])[:25])

# --- коллизии по чистому словарю
coll = collections.Counter()
for w, n in clean.most_common(30000):
    if not SPECIAL & set(w):
        continue
    for st in ['soft_h', 'soft_g', 'bare', 'e_style']:
        dd = degrade(w, st).replace('h', 'һ')
        if dd != w and dd in clean:
            coll[(w, dd)] = n
print('\nколлизий (пар слово->другое слово) среди 30k частых:', len(coll))
print('  частые:', [f'{a}→{b}' for (a, b), _ in coll.most_common(60)])
with open('collisions.tsv', 'w') as f:
    f.write('correct\tfreq\tdegraded_is_also_word\tfreq_other\n')
    for (a, b), n in coll.most_common():
        f.write(f'{a}\t{n}\t{b}\t{clean[b]}\n')

# --- правило «текст написан без спецбукв»
HARD_CHARS = set(LOOKALIKE) | set('hH')
def cyr_word_with_latin(tok):
    return bool(re.search('[а-яё]', tok, re.I)) and bool(re.search('[a-zA-Z' + ''.join(LOOKALIKE) + ']', tok))

TOKRE = re.compile(r"[\w-]+")
def text_flags(text):
    """Какие триггеры сработали. Возвращает dict имя->число."""
    raw_toks = TOKRE.findall(unicodedata.normalize('NFC', text))
    f = collections.Counter()
    f['latin/двойник в кириллическом слове'] = sum(cyr_word_with_latin(t) for t in raw_toks)
    norm = normalize_lookalikes(text).lower()
    toks = WORD.findall(norm)
    specials = sum(1 for ch in norm if ch in SPECIAL)
    f['_n'] = len(toks); f['_special'] = specials
    for w in toks:
        if w in clean:
            continue
        if re.search('ё|яя|юю', w):
            f['ё / яя / юю в незнакомом слове'] += 1
        cands = R.candidates(w)
        if cands and cands[0] != w and any(SPECIAL & set(v) for v in cands):
            f['незнакомое слово, восстановимое словарём'] += 1
    # буква, которой нет в тексте, хотя по длине её там ждут: ә ~6% букв
    letters = sum(ch.isalpha() for ch in norm)
    f['_letters'] = letters
    return f

def is_degraded(f):
    if f['latin/двойник в кириллическом слове']:
        return 'hard'
    if f['ё / яя / юю в незнакомом слове']:
        return 'hard'
    if f['_special'] == 0:
        if f['незнакомое слово, восстановимое словарём'] >= 1:
            return 'vocab'
        if f['_n'] >= 7:
            return 'length'
    else:
        # спецбуквы есть, но мало: частичная порча (нет ә на клавиатуре и т.п.)
        share = f['незнакомое слово, восстановимое словарём'] / max(1, f['_n'])
        if f['незнакомое слово, восстановимое словарём'] >= 2 and share >= 0.15:
            return 'partial'
    return ''

rnd = random.Random(3)
sample = rnd.sample(held, 4000)
print('\n== Правило «текст испорчен»: срабатывания')
fp = collections.Counter(); fp_ex = []
for t, nat, _ in sample:
    k = is_degraded(text_flags(t))
    fp[k] += 1
    if k and len(fp_ex) < 12:
        fp_ex.append((k, t[:90]))
print(f'  ЧИСТЫЕ предложения: помечено {100*(1-fp[""]/len(sample)):.2f}%  {dict(fp)}')
for e in fp_ex: print('    ', e)
by_len = collections.defaultdict(lambda: [0, 0])
for st in ['soft_h', 'soft_g', 'bare', 'e_style', 'russian', 'mixed'] + ['partial:' + p for p in PARTIAL]:
    c = collections.Counter(); bl = collections.defaultdict(lambda: [0, 0])
    for t, nat, _ in sample:
        if not SPECIAL & set(''.join(nat)):
            continue  # портить нечего
        if st.startswith('partial:'):
            keep = frozenset(SPECIAL - PARTIAL[st[8:]])
            dt = degrade_text(t, 'soft_h', rnd, keep)
            if dt == t:
                continue
        else:
            dt = degrade_text(t, st, rnd)
        k = is_degraded(text_flags(dt)); c[k] += 1
        b = min(len(nat), 8); bl[b][0] += 1; bl[b][1] += bool(k)
    tot = sum(c.values())
    print(f'  испорчено {st:18s}: поймано {100*(1-c[""]/tot):5.1f}%  {dict(c)}  по длине: ' +
          ' '.join(f'{b if b<8 else "8+"}:{100*y/x:.0f}%' for b, (x, y) in sorted(bl.items())))

# --- сквозное восстановление для помеченных текстов
def restore_text(text):
    norm = normalize_lookalikes(text)
    def rep(m):
        w = m.group(0); low = w.lower()
        cands = R.candidates(low)
        if not cands:
            return w
        best = cands[0]
        if w[:1].isupper():
            best = best[:1].upper() + best[1:]
        return best
    return re.sub(r"[А-Яа-яЁёӘәӨөҮүҺһҖҗҢң]+(?:-[А-Яа-яЁёӘәӨөҮүҺһҖҗҢң]+)*", rep, norm)

print('\n== Сквозная точность по словам после восстановления (текст уже помечен)')
for st in ['soft_h', 'bare', 'mixed']:
    ok = tot = 0; errs = collections.Counter()
    for t, nat, _ in sample[:1500]:
        dt = degrade_text(t, st, rnd)
        rt = restore_text(dt)
        a = WORD.findall(t.lower()); b = WORD.findall(rt.lower())
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            tot += 1; ok += x == y
            if x != y: errs[(y, x)] += 1
    print(f'  {st:8s}: {100*ok/tot:.1f}% слов верно;  ошибки: {[f"{y}(надо {x})" for (y, x), _ in errs.most_common(12)]}')

# --- что делает текущая модель с испорченным вводом
from core.transliterate import transliterate_word
words = [w for _, nat, _ in held for w in nat if SPECIAL & set(w)]
words = rnd.sample(sorted(set(words)), 1500)
same = {st: 0 for st in ['soft_h', 'bare']}
for w in words:
    ref = transliterate_word(w)
    for st in same:
        d = degrade(w, st).replace('h', 'һ')  # даже с правильной һ
        same[st] += transliterate_word(d) == ref
print('\n== Модель без исправления: доля слов со спецбуквами, у которых транслитерация испорченного ввода совпала с правильной')
for st, n in same.items():
    print(f'  {st}: {100*n/len(words):.1f}%')
print('  примеры:', [(w, transliterate_word(w), degrade(w, 'soft_h'), transliterate_word(degrade(w, 'soft_h').replace('h', 'һ'))) for w in words[:6]])
