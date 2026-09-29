# -*- coding: utf-8 -*-
"""Русский текст и русские слова: не должны считаться испорченным калмыцким."""
import collections, re, runpy, sys
sys.argv = ['eval2']
# переиспользуем словарь и правило из eval2 без его долгих хвостов
src = open('eval2.py').read().split("rnd = random.Random(3)")[0]
g = {}
exec(compile(src, 'eval2_head', 'exec'), g)
clean, R, WORD, text_flags, is_degraded = g['clean'], g['R'], g['WORD'], g['text_flags'], g['is_degraded']
import json, unicodedata
CORP = '/Users/vbadmaev1/Desktop/todorxoi-inference/corpus_out/corpus_sentences.jsonl'

# буква ж в родных словах
nat = collections.Counter(); ru = collections.Counter()
for line in open(CORP):
    r = json.loads(line)
    fw = {unicodedata.normalize('NFC', f['word'].lower()) for f in r['foreign_words'] if f['kind'] == 'ru_word'}
    for w in WORD.findall(unicodedata.normalize('NFC', r['text'].lower())):
        (ru if w in fw else nat)[w] += 1
tot = sum(nat.values())
zh = collections.Counter({w: n for w, n in nat.items() if 'ж' in w and 'җ' not in w})
print(f'«ж» в калмыцких токенах корпуса: {100*sum(zh.values())/tot:.2f}% токенов; частые: {[w for w,_ in zh.most_common(25)]}')
jj = sum(n for w, n in nat.items() if 'җ' in w)
print(f'для сравнения «җ»: {100*jj/tot:.2f}% токенов')

# русские слова из корпуса: сколько «восстановится» в калмыцкие
bad = collections.Counter()
for w, n in ru.items():
    if w in clean:
        continue
    c = R.candidates(w)
    if c and c[0] != w:
        bad[(w, c[0])] += n
print(f'\nрусские слова (ru_word), которые словарь «исправил» бы: {sum(bad.values())} из {sum(ru.values())} токенов;',
      [f'{a}->{b}' for (a, b), _ in bad.most_common(20)])

# русский текст: README и тексты бота
txt = open('/Users/vbadmaev1/Desktop/todorxoi-bot/README.md').read() + open('/Users/vbadmaev1/Desktop/todorxoi-bot/bot/texts.py').read()
sents = [s.strip() for s in re.split(r'[.!?\n]+', txt) if len(re.findall('[а-яё]+', s.lower())) >= 3]
k = collections.Counter(); ex = []
for s in sents:
    d = is_degraded(text_flags(s)); k[d] += 1
    if d and len(ex) < 10:
        f = text_flags(s)
        ex.append((d, s[:80]))
print(f'\nрусские предложения ({len(sents)}): помечены как испорченный калмыцкий {100*(1-k[""]/len(sents)):.1f}%  {dict(k)}')
for e in ex: print('   ', e)
fixes = collections.Counter()
for s in sents:
    for w in WORD.findall(s.lower()):
        if w not in clean:
            c = R.candidates(w)
            if c and c[0] != w:
                fixes[(w, c[0])] += 1
print('  русские слова, которые нашлись в калмыцком словаре после «восстановления»:', [f'{a}->{b}' for (a, b), _ in fixes.most_common(25)])
rw = [w for s in sents for w in WORD.findall(s.lower())]
print(f'  русских слов, которые сами есть в калмыцком словаре: {100*sum(w in clean for w in rw)/len(rw):.1f}%;',
      collections.Counter(w for w in rw if w in clean).most_common(20))
