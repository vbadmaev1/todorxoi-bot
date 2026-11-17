"""Compare exported models without dictionary lookup on the same held-out forms."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from core.hudum import ScriptConverter, _distance
from spelling import related_spelling


def evaluate(converter, rows, source, target):
    references = defaultdict(set)
    for row in rows:
        references[row[source]].add(row[target])
    words = sorted(references)
    correct = errors = chars = 0
    for offset in range(0, len(words), 64):
        batch = words[offset:offset + 64]
        for word, value in zip(batch, converter._predict(batch, target)):
            expected = references[word]
            correct += value in expected
            edits, length = min((_distance(value, variant), len(variant)) for variant in expected)
            errors += edits
            chars += length
    return {"headwords": len(words), "exact_match": correct / len(words), "cer": errors / chars}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    with (root / "pairs.tsv").open(encoding="utf-8") as handle:
        test = [row for row in csv.DictReader(handle, delimiter="\t") if row["split"] == "test"]
    spelling = [row for row in test if related_spelling(row["todo"], row["hudum"])]
    report = {}
    for path in args.models:
        converter = ScriptConverter(path)
        report[str(path)] = {
            dataset: {
                f"{source}_to_{target}": evaluate(converter, rows, source, target)
                for source, target in (("todo", "hudum"), ("hudum", "todo"))
            }
            for dataset, rows in (("all_test_pairs", test), ("related_test_pairs", spelling))
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
