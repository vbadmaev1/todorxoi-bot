# -*- coding: utf-8 -*-
"""
export_ocr_onnx.py — обученная модель распознавания (todo_ocr.pt из
ноутбука Kaggle/Colab) -> model/todo_ocr_int8.onnx для бота.

Запускается на машине разработчика, где есть torch, onnx и onnxruntime:

    python -m tools.export_ocr_onnx путь/к/todo_ocr.pt папка_со_страницами ... \\
        --out model/todo_ocr_int8.onnx

model_and_infer.py берётся из папки с todo_ocr.pt — у каждой модели свой.
Картинки — для калибровки int8: страницы или блоки с вертикальным письмом,
похожие на то, что будут присылать. Лучше смесь: 10–20 настоящих страниц
и 100–200 картинок из датасета (разные шрифты).

Что квантуется. На CPU ~80% времени — свёртки, ~20% — LSTM, поэтому
обычный quantize_dynamic (только LSTM и Linear) ускоряет всего на ~10%.
Здесь свёртки и Linear квантуются статически (QDQ, масштабы активаций — по
калибровочным столбцам), LSTM — динамически. Первая свёртка (вход —
картинка) и последний Linear (логиты для CTC) остаются fp32: они дешёвые,
а округление в них сильнее всего бьёт по точности. На модели от 2026-09-25:
16 МБ -> 4 МБ, в 3 раза быстрее на одном потоке, CER на синтетике тот же.

В конце печатается сверка int8 с fp32 на первых картинках.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from core.ocr import Model, column_line
from core.page_layout import split_page

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def export_fp32(pt_path: Path, out: Path) -> dict:
    sys.path.insert(0, str(pt_path.resolve().parent))
    from model_and_infer import load_model

    model, ck = load_model(str(pt_path))
    model.amp = False
    x = torch.rand(3, 1, ck["img_h"], 400)
    torch.onnx.export(
        model, x, str(out), input_names=["image"], output_names=["logits"], opset_version=17,
        dynamic_axes={"image": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "steps"}},
        dynamo=False,
    )
    m = onnx.load(str(out))
    # бот берёт алфавит и высоту строки прямо из модели
    m.metadata_props.add(key="chars", value=json.dumps(ck["chars"], ensure_ascii=False))
    m.metadata_props.add(key="img_h", value=str(ck["img_h"]))
    onnx.save(m, str(out))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for b, w in ((1, 128), (5, 1024)):         # другие batch и ширина, чем при экспорте
        x = torch.rand(b, 1, ck["img_h"], w)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        diff = np.abs(got - ref).max()
        assert got.shape == ref.shape and diff < 1e-3, f"ONNX не совпал с torch: {diff}"
    return ck


def batch_of(columns: list, img_h: int) -> np.ndarray:
    lines = [column_line(c, img_h) for c in columns]
    x = np.zeros((len(lines), 1, img_h, max(a.shape[1] for a in lines)), np.float32)
    for i, a in enumerate(lines):
        x[i, 0, :, :a.shape[1]] = a
    return x


class Columns(CalibrationDataReader):
    def __init__(self, batches):
        self.it = iter([{"image": b} for b in batches])

    def get_next(self):
        return next(self.it, None)


def quantize(fp32: Path, out: Path, batches: list) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pre, static = Path(tmp) / "pre.onnx", Path(tmp) / "static.onnx"
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
        g = onnx.load(str(pre)).graph
        convs = [n.name for n in g.node if n.op_type == "Conv"]
        matmuls = [n.name for n in g.node if n.op_type in ("MatMul", "Gemm")]
        quantize_static(
            str(pre), str(static), Columns(batches), quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=[convs[0], matmuls[-1]],
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
            # не копить активации всех батчей в памяти
            extra_options={"CalibMovingAverage": True, "CalibMaxIntermediateOutputs": 4},
        )
        quantize_dynamic(str(static), str(out), op_types_to_quantize=["LSTM"], weight_type=QuantType.QInt8)
    src, m = onnx.load(str(fp32)), onnx.load(str(out))
    for p in src.metadata_props:
        m.metadata_props.add(key=p.key, value=p.value)
    onnx.save(m, str(out))


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def image_paths(args):
    for a in map(Path, args):
        if a.is_dir():
            yield from sorted(p for p in a.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        else:
            yield a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", type=Path, help="todo_ocr.pt из архива модели")
    ap.add_argument("images", nargs="+", help="картинки или папки для калибровки int8")
    ap.add_argument("--out", type=Path, default=Path("model/todo_ocr_int8.onnx"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = args.out.with_name(args.out.stem.replace("_int8", "") + "_fp32.onnx")
    t = time.time()
    ck = export_fp32(args.pt, fp32)
    print(f"fp32: {fp32} ({fp32.stat().st_size / 1e6:.1f} МБ), выход совпадает с torch; {time.time() - t:.0f} с")

    pages = [c for c in (split_page(Image.open(p)) for p in image_paths(args.images)) if c]
    batches = [batch_of(cols[k:k + 5], ck["img_h"]) for cols in pages for k in range(0, len(cols), 5)]
    print(f"калибровка: {len(pages)} картинок, {sum(len(b) for b in batches)} столбцов")
    t = time.time()
    quantize(fp32, args.out, batches)
    print(f"int8: {args.out} ({args.out.stat().st_size / 1e6:.1f} МБ); {time.time() - t:.0f} с")

    a, b = Model(str(fp32), threads=1), Model(str(args.out), threads=1)
    diff = total = 0
    ta = tb = 0.0
    for cols in pages[:20]:
        t0 = time.time(); ra, _ = a.read(cols); t1 = time.time(); rb, _ = b.read(cols); t2 = time.time()
        ta += t1 - t0; tb += t2 - t1
        diff += sum(edit_distance(x, y) for x, y in zip(ra, rb)); total += sum(map(len, ra))
    n = max(1, min(20, len(pages)))
    print(f"int8 против fp32 (первые {n} картинок): расходится {100 * diff / max(1, total):.2f}% символов; "
          f"1 поток: fp32 {ta / n:.2f} с, int8 {tb / n:.2f} с на картинку")
    print(f"{fp32.name} боту не нужен — это промежуточный файл для сверки")


if __name__ == "__main__":
    main()
