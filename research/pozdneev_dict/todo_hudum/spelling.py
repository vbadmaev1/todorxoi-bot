"""Heuristic for separating related spellings from dictionary translations."""

from functools import lru_cache

# Broad sound classes for scoring only. Preserve the original Unicode in the dataset.
_GROUPS = (
    "ᠠᠡᡄ", "ᠢᡅ", "ᠣᠤᠥᠦᡆᡇᡈᡉ", "ᠧ", "ᠨ", "ᠩᡊ",
    "ᠪᡋᠫᡌᠹᡘᠸ", "ᠬᠭᡍᡎᡗᠺ", "ᠮᡏ", "ᠯ", "ᠰᠱ",
    "ᠲᠳᡐᡑ", "ᠴᠵᡒᡓᡔᡚᠼᠽᡜ", "ᠶᡕ", "ᠷ", "ᠾᡙ", "ᡛ", "ᡖ",
)
_CLASSES = {ch: i for i, group in enumerate(_GROUPS) for ch in group}
_CONTROLS = str.maketrans("", "", "\u180b\u180c\u180d\u180e\u180fᡃ")


@lru_cache(maxsize=65536)
def _skeleton(word):
    return tuple(_CLASSES.get(ch, ch) for ch in word.translate(_CONTROLS))


def spelling_distance(todo, hudum):
    a, b = _skeleton(todo), _skeleton(hudum)
    row = list(range(0, 4 * (len(b) + 1), 4))
    for i, x in enumerate(a, 1):
        previous, row = row, [4 * i]
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1 if x in (0, 1, 2, 3) and y in (0, 1, 2, 3) else 5
            row.append(min(row[-1] + 4, previous[j] + 4, previous[j - 1] + cost))
    return row[-1] / (4 * max(len(a), len(b), 1))


def related_spelling(todo, hudum):
    return todo.count(" ") == hudum.count(" ") and spelling_distance(todo, hudum) <= 0.29
