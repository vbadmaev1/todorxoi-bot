"""Adapter for the image2bichig CRNN exported to ONNX."""

import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .ocr import OcrUnavailable

_DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model" / "hudum_ocr_int8.onnx"
# image2bichig uses index 0 as CTC blank, 1 as space, then this Unicode order.
_CHARS = " " + "".join(map(chr, (
    *range(0x1800, 0x180F), *range(0x1810, 0x181A), *range(0x1820, 0x1879),
    *range(0x1880, 0x18AB), 0x202F,
)))
_LINE_HEIGHT = 32


class Model:
    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OcrUnavailable("не установлен onnxruntime") from exc
        if not path.is_file():
            raise OcrUnavailable(f"нет файла модели OCR худум: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = (
            int(os.environ.get("OCR_THREADS") or 0) or min(4, os.cpu_count() or 1)
        )
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "image" or inputs[0].shape[1:3] != [1, _LINE_HEIGHT]
                or len(outputs) != 1 or len(outputs[0].shape) != 3 or outputs[0].shape[-1] != len(_CHARS) + 1):
            raise OcrUnavailable("файл не соответствует ONNX-экспорту image2bichig")

    def read(self, columns: list, char_probs: bool = False):
        texts, probabilities = [], []
        conf_sum, conf_count = 0.0, 0
        for column in columns:
            line = column.convert("L").transpose(Image.Transpose.ROTATE_90)
            width = max(8, int(line.width * _LINE_HEIGHT / line.height))
            line = line.resize((width, _LINE_HEIGHT), Image.Resampling.BILINEAR)
            pixels = np.asarray(line, dtype=np.float32) / 255.0
            # Unlike Todo OCR, this graph expects white background = 1 and returns [time, batch, class].
            logits = self.session.run(None, {"image": pixels[None, None]})[0][:, 0]
            best = logits.argmax(-1)
            nonblank = best != 0
            keep = nonblank & np.concatenate(([True], best[1:] != best[:-1]))
            texts.append("".join(_CHARS[index - 1] for index in best[keep]))
            confidence = 1 / np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)
            probabilities.append(confidence[keep].tolist())
            conf_sum += float(confidence[nonblank].sum())
            conf_count += int(nonblank.sum())
        confidence = conf_sum / conf_count if conf_count else 0.0
        return (texts, confidence, probabilities) if char_probs else (texts, confidence)


@functools.lru_cache(maxsize=1)
def get_model() -> Model:
    path = Path(os.environ.get("OCR_HUDUM_MODEL_PATH") or _DEFAULT_MODEL).expanduser()
    return Model(path)
