"""Convert between Todo and traditional Mongolian script with a dictionary and ONNX."""

from collections import defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "todo_hudum"
_WORD = re.compile(r"[\u180b-\u180f\u1820-\u185c\u200c\u200d]+")
_LETTER = re.compile(r"[\u1820-\u185c]")
_TODO_ONLY = re.compile(r"[\u1843-\u185c]")
_JOINERS = str.maketrans("", "", "\u200c\u200d")
_VARIANTS = str.maketrans("", "", "\u180b\u180c\u180d\u180f")
BOS, EOS = 1, 2
_DIRECTIONS = {"hudum": 3, "todo": 4}


def _distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row = row, [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
    return row[-1]


class ScriptConverter:
    def __init__(self, path: Path):
        import onnxruntime as ort

        with gzip.open(path / "lexicon.json.gz", "rt", encoding="utf-8") as handle:
            bundle = json.load(handle)
        self.alphabet = bundle["alphabet"]
        self.ids = {char: i for i, char in enumerate(self.alphabet)}
        self.width = bundle["width"]
        self.allowed = {}
        self.lexicons = {target: defaultdict(set) for target in _DIRECTIONS}
        self.variants = {target: defaultdict(set) for target in _DIRECTIONS}
        for todo, hudum in bundle["pairs"]:
            for target, source, result in (("hudum", todo, hudum), ("todo", hudum, todo)):
                self.lexicons[target][source].add(result)
                self.variants[target][source.translate(_VARIANTS)].add(result)
        for target, chars in bundle["allowed"].items():
            mask = np.zeros(len(self.alphabet), dtype=bool)
            mask[[EOS, *(self.ids[char] for char in chars)]] = True
            self.allowed[target] = mask

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(path / "encoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(path / "decoder.onnx"), opts, providers=["CPUExecutionProvider"])

    def _predict(self, words: list[str], target: str) -> list[str]:
        src = np.zeros((len(words), self.width), dtype=np.int64)
        for i, word in enumerate(words):
            src[i, :len(word) + 2] = [_DIRECTIONS[target], *(self.ids[ch] for ch in word), EOS]
        memory, keys, mask, state = self.encoder.run(None, {"src": src})
        tokens = np.full((len(words), 1), BOS, dtype=np.int64)
        done = np.zeros(len(words), dtype=bool)
        out = [[] for _ in words]
        for _ in range(self.width - 1):
            logits, state = self.decoder.run(None, {
                "tokens": tokens, "memory": memory, "keys": keys, "mask": mask, "state": state,
            })
            tokens = np.where(self.allowed[target], logits, -np.inf).argmax(-1).astype(np.int64)
            for i, idx in enumerate(tokens[:, 0]):
                if not done[i] and idx != EOS:
                    out[i].append(self.alphabet[idx])
            done |= tokens[:, 0] == EOS
            if done.all():
                break
        # A result with no EOS or letters is an unsuccessful decode.
        return [
            word if finished and _LETTER.search(word) else ""
            for word, finished in zip(map("".join, out), done)
        ]

    def convert(self, text: str, target: str) -> str:
        """Convert dictionary phrases and individual words, preserving other text."""
        if target not in _DIRECTIONS:
            raise ValueError("Unknown target script")
        lexicon, variants = self.lexicons[target], self.variants[target]
        parts = re.split(f"({_WORD.pattern})", text)
        items = {}
        i = 1
        while i < len(parts):
            first = parts[i]
            i_next = i + 2
            # Prefer recorded phrases when words are separated by ordinary single spaces.
            phrase = first
            for j in range(i + 2, len(parts), 2):
                if parts[j - 1] != " ":
                    break
                phrase += " " + parts[j]
                if len(phrase) > self.width - 2:
                    break
                if phrase.translate(_JOINERS) in lexicon:
                    i_next = j + 2
            raw = "".join(parts[i:i_next - 1])
            key = unicodedata.normalize("NFC", raw).translate(_JOINERS)
            if target == "hudum" and key not in lexicon:
                key = key.replace("ᡚ", "ᡓ")
            if _LETTER.search(key) and not (target == "todo" and _TODO_ONLY.search(key)):
                items[i] = (i_next, key)
            i = i_next

        if not items:
            raise ValueError("No words in the source script")
        results = {}
        predict = []
        options = {}
        for word in dict.fromkeys(word for _, word in items.values()):
            candidates = lexicon.get(word) or variants.get(word.translate(_VARIANTS), set())
            if len(candidates) == 1:
                results[word] = next(iter(candidates))
                continue
            key = "".join(ch for ch in word if ch in self.ids)
            if key and len(key) <= self.width - 2:
                options[word] = candidates
                predict.append((word, key))
        for offset in range(0, len(predict), 64):
            batch = predict[offset:offset + 64]
            predictions = self._predict([key for _, key in batch], target)
            for (word, _), value in zip(batch, predictions):
                candidates = options[word]
                if candidates:
                    value = min(candidates, key=lambda candidate: (_distance(value, candidate), candidate))
                if value:
                    results[word] = value
        for start, (end, word) in items.items():
            if word in results:
                parts[start:end - 1] = [results[word]] + [""] * (end - start - 2)
        return "".join(parts)


@lru_cache(maxsize=1)
def get_converter() -> ScriptConverter:
    return ScriptConverter(MODEL_DIR)
