import json, re, collections, unicodedata, pickle, numpy as np
CORP='/Users/vbadmaev1/Desktop/todorxoi-inference/corpus_out/corpus_sentences.jsonl'
NPZ='/Users/vbadmaev1/Desktop/todorxoi-bot/model/translit_model.npz'
WORD=re.compile(r"[а-яёәөүһҗң]+(?:-[а-яёәөүһҗң]+)*")
freq=collections.Counter(); foreign=collections.Counter(); sents=[]
for l in open(CORP):
    r=json.loads(l)
    if r['ocr_suspect']: continue
    fw={unicodedata.normalize('NFC',f['word'].lower()) for f in r['foreign_words']}
    t=unicodedata.normalize('NFC',r['text'].lower())
    toks=WORD.findall(t)
    for w in toks:
        (foreign if w in fw else freq)[w]+=1
    sents.append((r['text'], [w for w in toks if w not in fw], [w for w in toks if w in fw]))
d=np.load(NPZ); meta=json.loads(bytes(d['meta_json']).decode())
dic=set(unicodedata.normalize('NFC',k) for k in meta['dictionary'])
pickle.dump(dict(freq=freq,foreign=foreign,sents=sents,dic=dic),open('vocab.pkl','wb'))
SP=set('әөүһҗң')
tok=sum(freq.values()); tok_sp=sum(c for w,c in freq.items() if SP&set(w))
print('sentences',len(sents),'kalmyk tokens',tok,'types',len(freq),'dict',len(dic))
print('share tokens with special letters %.1f%%'%(100*tok_sp/tok))
print('types with special %.1f%%'%(100*sum(1 for w in freq if SP&set(w))/len(freq)))
lc=collections.Counter()
for w,c in freq.items():
    for ch in w: lc[ch]+=c
tot=sum(lc.values())
print(' '.join(f'{ch}:{100*lc[ch]/tot:.2f}' for ch in 'әөүһҗңяюёэеаоуиы'))
print('top words without special letters:', [w for w,_ in freq.most_common(300) if not SP&set(w)][:120])
