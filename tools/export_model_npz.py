# -*- coding: utf-8 -*-
"""
export_model_npz.py — вынуть веса из .pt-чекпойнта в .npz, чтобы боту
больше не был нужен torch.

Запускается один раз, на машине разработчика, где torch есть:

    python -m tools.export_model_npz model/translit_model_cyr2lat.pt \\
        --out model/translit_model.npz

В .npz кладём сами веса (float32) и рядом, отдельным массивом байт, json
со словарями символов и точным словарём слов. Так весь артефакт остаётся
одним файлом, который просто лежит в репозитории.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

META_KEY = "meta_json"


def export(pt_path: Path, npz_path: Path) -> None:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    arrays = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in ckpt["model_state"].items()
    }

    meta = {
        "src_vocab": ckpt["src_vocab"],
        "tgt_vocab": ckpt["tgt_vocab"],
        "emb_dim": int(ckpt["emb_dim"]),
        "hid_dim": int(ckpt["hid_dim"]),
        "max_len": int(ckpt.get("max_len", 32)),
        "dictionary": ckpt.get("dictionary", {}),
    }
    raw = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    arrays[META_KEY] = np.frombuffer(raw, dtype=np.uint8)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    params = sum(a.size for n, a in arrays.items() if n != META_KEY)
    print(f"Записан {npz_path}")
    print(f"  параметров: {params}")
    print(f"  слов в словаре: {len(meta['dictionary'])}")
    print(f"  размер файла: {npz_path.stat().st_size / 1e6:.2f} МБ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=Path("model/translit_model.npz"))
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
