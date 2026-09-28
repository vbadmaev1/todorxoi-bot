# -*- coding: utf-8 -*-
"""
check_numpy_matches_torch.py — сверить NumPy-реализацию с исходной
torch-моделью на всех словах, какие есть.

Смысл проверки: ошибка в переносе формул GRU или внимания не падает с
исключением. Сеть продолжит работать и выдавать правдоподобные слова —
просто другие. Поймать это можно только сравнением выходов, причём не на
десятке примеров, а на всём объёме.

Скрипт нужен один раз, на машине разработчика, где torch ещё стоит:

    python -m tools.check_numpy_matches_torch \\
        --pt model/translit_model_cyr2lat.pt \\
        --npz model/translit_model.npz

Сравниваются строки на выходе, а не числа: именно строки видит человек,
и именно на них расхождение имеет значение. Расхождение в последнем
знаке после запятой, не поменявшее argmax, нам безразлично.

Веса в .npz хранятся в float16, поэтому единичные расхождения ожидаемы
(на модели 2026-09-29 — единицы на 37 тысяч слов). Ошибка в формулах
дала бы расхождение на большой доле слов, так что проверка падает, только
если расходится больше --max-share (по умолчанию 0.1 %).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", type=Path, default=Path("model/translit_model_cyr2lat.pt"))
    ap.add_argument("--npz", type=Path, default=Path("model/translit_model.npz"))
    ap.add_argument("--limit", type=int, default=0, help="взять только N слов")
    ap.add_argument("--max-share", type=float, default=0.001,
                    help="допустимая доля расхождений (float16 даёт единичные)")
    args = ap.parse_args()

    import torch  # noqa: F401  — нужен только здесь, боту он больше не нужен

    from core.model_numpy import load as load_np
    from tools._torch_reference import load_torch_reference

    ref = load_torch_reference(args.pt)
    npm = load_np(args.npz)

    words = sorted(ref["dictionary"].keys())
    if args.limit:
        words = words[: args.limit]
    # словарь — это как раз те слова, которые бот отдаёт НЕ модели, поэтому
    # к ним добавляем заведомо незнакомые: именно на них работает сеть
    unseen = [
        "тертцхн", "медвч", "цуглрад", "хәрүлҗ", "оньдин", "заагтан",
        "күцәмҗтә", "шидрхн", "бәәршлҗ", "олзлгдсн", "төрскндән",
    ]
    words += [w for w in unseen if w not in ref["dictionary"]]

    print(f"Сверяю {len(words)} слов...")
    mismatches = []
    t0 = time.perf_counter()
    for i, w in enumerate(words, 1):
        a = ref["translate"](w)
        b = npm.translate_word(w)
        if a != b:
            mismatches.append((w, a, b))
        if i % 2000 == 0:
            print(f"  {i}/{len(words)}...")
    took = time.perf_counter() - t0

    print(f"\nПроверено: {len(words)} слов за {took:.1f} с")
    if not mismatches:
        print("Расхождений нет: NumPy повторяет torch слово в слово ✓")
        return
    share = len(mismatches) / len(words)
    print(f"Расхождений: {len(mismatches)} ({share:.3%})")
    for w, a, b in mismatches[:20]:
        print(f"   {w:20s} torch={a!r:24s} numpy={b!r}")
    if share > args.max_share:
        print(f"Больше допустимых {args.max_share:.3%} — похоже на ошибку в формулах")
        sys.exit(1)
    print("В пределах погрешности float16 ✓")


if __name__ == "__main__":
    main()
