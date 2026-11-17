"""Train a shared character encoder/decoder in the two directions."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from spelling import related_spelling

PAD, BOS, EOS, TO_HUDUM, TO_TODO = range(5)
WIDTH = 34  # direction marker, up to 32 source characters, EOS
HIDDEN = 128


class Converter(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.emb = nn.Embedding(size, 80, padding_idx=PAD)
        self.encoder = nn.GRU(80, HIDDEN, batch_first=True, bidirectional=True)
        self.initial = nn.Linear(2 * HIDDEN, HIDDEN)
        self.decoder = nn.GRU(80, HIDDEN, batch_first=True)
        self.keys = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.head = nn.Sequential(
            nn.Linear(3 * HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, size)
        )

    def encode(self, src):
        memory, state = self.encoder(self.emb(src))
        return memory, self.keys(memory), src.ne(PAD), self.initial(
            torch.cat((state[0], state[1]), dim=-1)
        ).tanh().unsqueeze(0)

    def decode(self, tokens, memory, keys, mask, state):
        query, state = self.decoder(self.emb(tokens), state)
        scores = (query @ keys.transpose(1, 2)) / HIDDEN ** 0.5
        attention = scores.masked_fill(~mask.unsqueeze(1), -1e4).softmax(-1)
        return self.head(torch.cat((query, attention @ memory), dim=-1)), state

    def forward(self, src, tokens):
        return self.decode(tokens, *self.encode(src))[0]


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        return self.model.encode(src)


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, memory, keys, mask, state):
        return self.model.decode(tokens, memory, keys, mask, state)


def export(model, src, out):
    model = model.cpu().eval()
    src = src[:2].cpu()
    with torch.no_grad():
        context = model.encode(src)
    torch.onnx.export(
        Encoder(model), (src,), out / "encoder.onnx", dynamo=False, opset_version=17,
        input_names=["src"], output_names=["memory", "keys", "mask", "state"],
        dynamic_axes={"src": {0: "batch"}, "memory": {0: "batch"}, "keys": {0: "batch"},
                      "mask": {0: "batch"}, "state": {1: "batch"}},
    )
    torch.onnx.export(
        Decoder(model), (torch.full((2, 1), BOS), *context),
        out / "decoder.onnx", dynamo=False, opset_version=17,
        input_names=["tokens", "memory", "keys", "mask", "state"],
        output_names=["logits", "next_state"],
        dynamic_axes={"tokens": {0: "batch"}, "memory": {0: "batch"}, "keys": {0: "batch"},
                      "mask": {0: "batch"}, "state": {1: "batch"},
                      "logits": {0: "batch"}, "next_state": {1: "batch"}},
    )


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row = row, [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
    return row[-1]


@torch.inference_mode()
def evaluate(model, directions, encode, alphabet, allowed, device, batch_size):
    model.eval()
    report = {}
    for direction, references in directions.items():
        vocab_mask = allowed[direction]
        words = sorted(references)
        exact = edits = chars = 0
        examples = []
        for offset in range(0, len(words), batch_size):
            batch = words[offset:offset + batch_size]
            ids = torch.tensor([encode(word, direction) for word in batch], device=device)
            memory, keys, mask, state = model.encode(ids)
            token = torch.full((len(batch), 1), BOS, device=device)
            done = torch.zeros(len(batch), device=device, dtype=torch.bool)
            sequences = []
            for _ in range(WIDTH - 1):
                logits, state = model.decode(token, memory, keys, mask, state)
                token = logits.masked_fill(~vocab_mask, -1e4).argmax(-1)
                done |= token[:, 0].eq(EOS)
                sequences.append(token[:, 0])
                if done.all():
                    break
            for word, seq in zip(batch, torch.stack(sequences, dim=1).tolist()):
                result = "".join(alphabet[idx] for idx in seq[:seq.index(EOS)]) if EOS in seq else ""
                targets = references[word]
                edits_for_word, length = min((distance(result, t), len(t)) for t in targets)
                exact += result in targets
                edits += edits_for_word
                chars += length
                if len(examples) < 24 and len(word) >= 5:
                    examples.append({"input": word, "prediction": result, "references": sorted(targets)})
        report[direction] = {
            "headwords": len(words), "exact_match": exact / len(words),
            "cer": edits / chars, "examples": examples,
        }
    return report


def train(data, out, epochs, batch_size, all_pairs=False):
    out.mkdir(parents=True, exist_ok=True)
    random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"training on {device}", flush=True)
    with data.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    clean_rows = [row for row in rows if related_spelling(row["todo"], row["hudum"])]
    train_rows = [row for row in (rows if all_pairs else clean_rows) if row["split"] == "train"]
    print(f"{len(train_rows)} training pairs, {len(clean_rows)} spelling pairs overall", flush=True)
    script_chars = {
        script: sorted({ch for row in train_rows for ch in row[script]})
        for script in ("todo", "hudum")
    }
    # Include the script inventories in the vocabulary so uncommon test characters can be encoded.
    alphabet = ["<pad>", "<bos>", "<eos>", "<hudum>", "<todo>"] + sorted(
        set(" \u180b\u180c\u180d\u180e\u180f\u202f")
        | set(map(chr, range(0x1820, 0x185d)))
    )
    ids = {ch: i for i, ch in enumerate(alphabet)}

    def padded(seq, width):
        return seq + [PAD] * (width - len(seq))

    def encode(word, direction):
        return padded([direction, *(ids[ch] for ch in word), EOS], WIDTH)

    x, y = [], []
    allowed = {}
    eval_sets = {split: {} for split in ("validation", "test")}
    directions = ((TO_HUDUM, "todo", "hudum"), (TO_TODO, "hudum", "todo"))
    for direction, source, target in directions:
        mask = torch.zeros(len(ids), device=device, dtype=torch.bool)
        mask[[EOS, *(ids[ch] for ch in script_chars[target])]] = True
        allowed[direction] = mask
        for row in train_rows:
            x.append(encode(row[source], direction))
            y.append(padded([BOS, *(ids[ch] for ch in row[target]), EOS], WIDTH))
        for split, lookup in eval_sets.items():
            references = defaultdict(set)
            for row in clean_rows:
                if row["split"] == split:
                    references[row[source]].add(row[target])
            lookup[direction] = references
    x = torch.tensor(x, device=device)
    y = torch.tensor(y, device=device)
    model = Converter(len(ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0002)
    best_score = -1
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in torch.randperm(len(x), device=device).split(batch_size):
            src, tgt = x[batch], y[batch]
            # Trim trailing decoder padding to the longest target in this batch.
            tgt = tgt[:, :int(tgt.ne(PAD).sum(1).max())]
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), tgt[:, 1:].reshape(-1), ignore_index=PAD)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        valid = evaluate(model, eval_sets["validation"], encode, alphabet, allowed, device, batch_size)
        score = sum(v["exact_match"] for v in valid.values()) / len(valid)
        metrics = {"epoch": epoch, "loss": torch.stack(losses).mean().item(),
                   "validation": {k: {a: b for a, b in v.items() if a != "examples"} for k, v in valid.items()}}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    test = evaluate(model, eval_sets["test"], encode, alphabet, allowed, device, batch_size)
    names = {TO_HUDUM: "todo_to_hudum", TO_TODO: "hudum_to_todo"}
    metrics = {"device": str(device), "training_pairs": len(train_rows),
               "lexicon_pairs": len(clean_rows), "best_epoch": best_epoch, "history": history,
               "test": {names[k]: v for k, v in test.items()}}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({names[k]: {a: b for a, b in v.items() if a != "examples"} for k, v in test.items()}), flush=True)
    export(model, x, out)
    # Production can look up compatible dictionary forms. Metrics above use only the model.
    lexicon = [(row["todo"], row["hudum"]) for row in clean_rows]
    bundle = {"alphabet": alphabet, "width": WIDTH, "allowed": script_chars, "pairs": lexicon}
    (out / "lexicon.json.gz").write_bytes(gzip.compress(json.dumps(bundle, ensure_ascii=False).encode(), mtime=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--all-pairs", action="store_true", help="Train on all pairs for a baseline")
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.batch_size, args.all_pairs)
