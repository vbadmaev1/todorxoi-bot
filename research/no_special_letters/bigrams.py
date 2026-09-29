# какие буквосочетания встречаются в правильном калмыцком (родные слова) и в русских вкраплениях
import pickle, collections
V = pickle.load(open('vocab.pkl', 'rb')); freq = V['freq']; foreign = V['foreign']
CONS = set('бвгджзйклмнпрстфхцчшщһҗң')


def pat(w):
    out = []
    for i, ch in enumerate(w):
        prev = w[i - 1] if i else '^'
        if ch in 'яюё':
            kind = 'C+' if prev in CONS else ('^' if prev == '^' else ('ъь' if prev in 'ъь' else 'V+'))
            out.append(kind + ch)
    return out


for name, src in [('native', freq), ('russian-in-corpus', foreign)]:
    c = collections.Counter(); ex = collections.defaultdict(collections.Counter); tot = sum(src.values())
    for w, n in src.items():
        for p in set(pat(w)):
            c[p] += n; ex[p][w] += n
    print(f'== {name} tokens={tot}')
    for p, n in c.most_common():
        print(f'  {p:6s} {n:6d} ({1e4*n/tot:.1f} на 10k) e.g. {[w for w,_ in ex[p].most_common(12)]}')
for name, src in [('native', freq), ('russian', foreign)]:
    for dd in ['яя', 'юю', 'ёё', 'аа', 'оо', 'уу', 'ээ', 'ее', 'ии']:
        sub = collections.Counter({w: c for w, c in src.items() if dd in w})
        print(name, dd, sum(sub.values()), [w for w, _ in sub.most_common(8)])
