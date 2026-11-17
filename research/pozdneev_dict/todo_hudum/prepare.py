"""Extract and split paired spellings from the August XLSX export."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import csv
import json
import re
import unicodedata

from openpyxl import load_workbook

from spelling import related_spelling

TODO = re.compile(r"[\u1820-\u185c\u180b-\u180f ]+")
HUDUM = re.compile(r"[\u1820-\u1842\u180b-\u180f ]+")
CONTROLS = str.maketrans("", "", "\u180b\u180c\u180d\u180e\u180f")
NOTES = re.compile(r"\s*[（(]\s*[～~][^）)]*[）)]\s*$")
MAX_LENGTH = 32


def clean(value):
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"[\r\n\t \u00a0]+", " ", value).strip()


def read_pairs(path):
    stats = Counter()
    pairs = set()
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(book.active.values)
        if next(rows) != ("todo", "xudam"):
            raise ValueError("Expected columns todo, xudam")
        for todo, hudum in rows:
            stats["raw"] += 1
            if not isinstance(todo, str) or not isinstance(hudum, str):
                stats["missing"] += 1
                continue
            todo, notes = NOTES.subn("", clean(todo))
            hudum = clean(hudum)
            stats["removed_cross_reference"] += notes
            if not TODO.fullmatch(todo) or not HUDUM.fullmatch(hudum):
                stats["non_spelling"] += 1
                continue
            if not todo.translate(CONTROLS).strip() or not hudum.translate(CONTROLS).strip():
                stats["non_spelling"] += 1
                continue
            if max(len(todo), len(hudum)) > MAX_LENGTH:
                stats["too_long"] += 1
                continue
            if (todo, hudum) in pairs:
                stats["duplicates"] += 1
            pairs.add((todo, hudum))
    finally:
        book.close()
    return sorted(pairs), stats


def split_pairs(pairs):
    # Shared headwords and their Unicode variants stay in one split in both directions.
    parent = {}

    def root(key):
        while key in parent:
            key = parent[key]
        return key

    def keys(pair):
        return tuple((i, word.translate(CONTROLS)) for i, word in enumerate(pair))

    for pair in pairs:
        a, b = map(root, keys(pair))
        if a != b:
            parent[max(a, b)] = min(a, b)

    groups = defaultdict(list)
    for pair in pairs:
        groups[root(keys(pair)[0])].append(pair)

    for key, group in sorted(groups.items()):
        bucket = sha256(f"todo-hudum-v1:{key}".encode()).digest()[0]
        split = "test" if bucket < 26 else "validation" if bucket < 52 else "train"
        for todo, hudum in group:
            yield split, todo, hudum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs, stats = read_pairs(args.source)
    splits = list(split_pairs(pairs))
    spellings = [row for row in splits if related_spelling(*row[1:])]
    for filename, rows in (("pairs.tsv", splits), ("spelling_pairs.tsv", spellings)):
        with (args.out / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["split", "todo", "hudum"])
            writer.writerows(rows)
    summary = {
        "source": args.source.name,
        "sha256": sha256(args.source.read_bytes()).hexdigest(),
        "rows": dict(stats),
        "split_pairs": dict(Counter(row[0] for row in splits)),
        "related_spelling_pairs": dict(Counter(row[0] for row in spellings)),
        "ambiguous_todo": sum(n > 1 for n in Counter(a for a, _ in pairs).values()),
        "ambiguous_hudum": sum(n > 1 for n in Counter(b for _, b in pairs).values()),
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
