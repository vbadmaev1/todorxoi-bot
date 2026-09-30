# -*- coding: utf-8 -*-
"""Замер боевого модуля core/fix_letters.py на двух семплах.

Словарь строится тем же tools/build_letters_vocab.py, но без отложенных
книг. Из эталона убраны предложения с явным OCR-шумом (латиница, «ё»,
«э» не в начале слова): они сами написаны неправильно, и честно мерить на
них нельзя (см. «менгн», «темдгтэhер» в первой версии examples.tsv).

    python eval9.py            # оба семпла
    COST_WEIGHT=1 python eval9.py
"""
import collections, json, os, random, re, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import core.fix_letters as FL
from core.fix_letters import LetterFixer, SPECIAL, _FOREIGN
from tools.build_letters_vocab import build_vocab, _NEVER
from degrade import PARTIAL, degrade_text

if os.environ.get('COST_WEIGHT'):
    FL.COST_WEIGHT = float(os.environ['COST_WEIGHT'])
CORP = Path.home() / 'Desktop/todorxoi-inference/corpus_out/corpus_sentences.jsonl'
NPZ = ROOT / 'model/translit_model.npz'
WORD = re.compile(r"[а-яёәөүһҗң]+(?:-[а-яёәөүһҗң]+)*")
SAMPLES = [
    ('семпл 1', {'бадмин_үкрчә_теегин_республик', 'эрнҗана_ботхн_һалан_хадһл'}, 7),
    ('семпл 2', {'хуучн_үгд_худл_уга', 'балакан_а_элст_деер_мандлсн_одн'}, 42),
]
RU_TEXT = [s.strip() for s in re.split(r'[.!?\n]+', open(ROOT / 'README.md').read() + open(ROOT / 'bot/texts.py').read())
           if len(re.findall('[а-яё]+', s.lower())) >= 3]


def noisy(text):
    return bool(_FOREIGN.search(text)) or any(_NEVER.search(w) for w in WORD.findall(text.lower()))


for name, held_books, seed in SAMPLES:
    vocab, protected, _, trusted = build_vocab(CORP, NPZ, exclude_books=held_books)
    fx = LetterFixer(vocab, protected, trusted)
    held = []
    for line in open(CORP):
        r = json.loads(line)
        if r['book_id'] in held_books and not r['ocr_suspect']:
            held.append(unicodedata.normalize('NFC', r['text']))
    n_all = len(held)
    held = [t for t in held if not noisy(t)]
    rnd = random.Random(seed)
    sample = rnd.sample(held, 3000)
    print(f'\n===== {name}: {", ".join(sorted(held_books))}; словарь {len(vocab)}; '
          f'предложений {n_all}, с явным шумом убрано {n_all - len(held)}')

    # правильные
    flagged = 0; changed = collections.Counter(); toks = 0; why = collections.Counter(); why_ex = collections.defaultdict(list)
    for t in sample:
        r = fx.fix(t)
        flagged += bool(r.detection.flag)
        if r.detection.flag:
            why[r.detection.flag] += 1
            if len(why_ex[r.detection.flag]) < 4:
                why_ex[r.detection.flag].append(t[:80])
        toks += len(WORD.findall(t.lower()))
        changed.update(r.fixes)
    real = {k: n for k, n in changed.items() if k[0].lower() in vocab}
    print(f'  правильные: помечено {100*flagged/len(sample):.2f}%, изменено слов {sum(changed.values())} '
          f'({100*sum(changed.values())/toks:.3f}%), словарных {sum(real.values())}')
    print('    ', [f'{a}→{b}' for (a, b), _ in changed.most_common(20)])
    print('     почему помечены:', dict(why))
    for k, v in why_ex.items():
        for e in v:
            print(f'       [{k}] {e}')
    # явная проверка (режим /fix): незнакомые слова правятся и в правильном тексте
    forced = collections.Counter()
    for t in sample:
        forced.update(fx.fix(t, force=True).fixes)
    print(f'  правильные, режим /fix: изменено слов {sum(forced.values())} ({100*sum(forced.values())/toks:.3f}%)')
    print('    ', [f'{a}→{b}' for (a, b), _ in forced.most_common(30)])
    rf = sum(bool(fx.detect(s).flag) for s in RU_TEXT)
    print(f'  русский текст: помечено {100*rf/len(RU_TEXT):.1f}%')

    # обнаружение
    line = []
    for st in ['soft_h', 'soft_g', 'bare', 'e_style', 'mixed'] + ['partial:' + p for p in PARTIAL]:
        r2 = random.Random(seed); hit = n = 0
        for t in sample[:1500]:
            keep = frozenset(SPECIAL - PARTIAL[st[8:]]) if st.startswith('partial:') else frozenset()
            d = degrade_text(t, 'soft_h' if keep else st, r2, keep)
            if d == t:
                continue
            n += 1; hit += bool(fx.detect(d).flag)
        line.append(f'{st.replace("partial:", "p:")} {100*hit/n:.1f}%')
    print('  поймано:', '  '.join(line))

    # точность и опечатки
    line = []; errs = collections.Counter()
    for st in ['soft_h', 'soft_g', 'bare', 'e_style', 'mixed']:
        r2 = random.Random(seed + 1); ok = tot = 0
        for t in sample[:1500]:
            d = degrade_text(t, st, r2); r = fx.fix(d)
            a, b = WORD.findall(t.lower()), WORD.findall(r.text.lower())
            if len(a) != len(b):
                continue
            tot += len(a); ok += sum(x == y for x, y in zip(a, b))
            for x, y in zip(a, b):
                if x != y:
                    errs[(y, x)] += 1
        line.append(f'{st} {100*ok/tot:.1f}%')
    print('  точность по словам:', '  '.join(line))
    print('  частые ошибки (стало / надо):', [f'{y}/{x}' for (y, x), _ in errs.most_common(20)])
