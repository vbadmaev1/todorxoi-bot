# -*- coding: utf-8 -*-
"""
ocr.py — распознавание тодо бичиг с фото: картинка -> столбцы -> текст.

  1. page_layout.split_page режет страницу на столбцы (выпрямляет наклон,
     выкидывает колонтитулы и пыль) — подробности в самом модуле.
  2. CRNN + CTC (model/todo_ocr_int8.onnx) читает каждый столбец: столбец
     поворачивается, приводится к высоте 64 и идёт в сеть. Модель int8 под
     onnxruntime: 4 МБ, torch не нужен; на CPU ~1 с на полную страницу в
     один поток. Как она получена из обученной — tools/export_ocr_onnx.py.
  3. Транслитерация — тем же todo_to_translit, что и в текстовых режимах;
     перед ней убираются висячие узкие пробелы, знаки — в латинские.
  4. Картинка-проверка: выпрямленный исходник с рамками и номерами
     столбцов. Видно, как бот разрезал страницу, и какая строка ответа
     какому столбцу соответствует. Стоит ~50 мс и один JPEG.

Всё синхронное и счётное — в боте звать через asyncio.to_thread и по одной
картинке за раз (см. bot/handlers/ocr.py): модель с библиотеками держит
~95 МБ, полная страница — ещё ~160 МБ на время чтения, две параллельно —
вдвое больше.
"""

import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .punctuation import DEFAULT_PUNCT, apply_punctuation
from .translit_todo import normalize_j, todo_to_translit

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL = _HERE.parent / "model" / "todo_ocr_int8.onnx"

# Больше этого по длинной стороне картинка уменьшается: память разметки
# растёт с площадью, а точность на сканах одинаковая от 1000 до 2700 px.
MAX_SIDE = 2500
# Столбцов в сеть за один прогон: 2 — пик ~280 МБ на страницу, 8 — ~670 МБ
# при той же скорости и точности.
MAX_BATCH = 2
# Картинка-проверка крупнее не нужна: Telegram всё равно пережмёт.
OVERLAY_SIDE = 1600
# Средняя уверенность модели (вероятность выбранного символа по всем
# непустым шагам CTC) ниже этого — вероятно, на фото не тодо бичиг или
# текст плохо читается. Замерено: страницы тодо бичиг у Позднеева 0.95–0.97
# (и так же после сжатия до 1280 px в JPEG), синтетика 0.96–1.00; русские
# страницы той же книги — почти всегда ни одной буквы, а где что-то нашлось,
# 0.43–0.93 (0.93 — одна буква на всю страницу, её ловит MIN_LETTERS).
LOW_CONFIDENCE = 0.9
# Меньше букв на всю картинку — считаем, что тодо бичиг на ней нет.
MIN_LETTERS = 2

_BOX_COLORS = ((230, 40, 40), (30, 110, 235))   # соседние столбцы — разным цветом

_GAP = " ᠂᠃︱︖︕"                  # пробел и знаки препинания из алфавита модели
_NOT_LETTERS = set(_GAP + "\u202f")
# Узкий неразрывный пробел (U+202F) отделяет суффикс и стоит между буквами.
# Рядом с пробелом или знаком он смысла не имеет, а модель ставит его на
# широких промежутках между словами (в книгах столбцы выключены по высоте,
# и промежутки растянуты). В транслитерации он стал бы висячим дефисом:
# «talaa- ǰiliyaiǰi-».
_LOOSE_NNBSP = re.compile(f"\u202f+(?=[{_GAP}]|$)|(?<=[{_GAP}])\u202f+|^\u202f+")
# На странице знаки препинания — настоящие монгольские (᠂ ᠃), а
# todo_to_translit знает только вертикальные формы из текстового режима
# (︐ ︒). В транслитерации — обычные латинские знаки. У бирги латинской
# пары нет, четыре точки — конец текста, то есть точка.
_PUNCT_TO_LATIN = str.maketrans(
    {"᠂": ",", "᠃": ".", "︖": "?", "︕": "!", "︱": "—", "᠀": None, "᠅": "."}
)


def tidy(text: str) -> str:
    text = _LOOSE_NNBSP.sub("", text)
    text = re.sub("\u202f{2,}", "\u202f", text)
    return re.sub(" {2,}", " ", text).strip(" ")


def to_translit(todo: str) -> str:
    return todo_to_translit(todo).translate(_PUNCT_TO_LATIN)


class OcrError(Exception):
    """Ошибка, текст которой можно показать пользователю как есть."""


class OcrUnavailable(OcrError):
    """Нет onnxruntime или файла модели: бот работает, но фото не читает."""


@dataclass
class OcrResult:
    columns: list                          # тодо бичиг, по строке на столбец
    translit_columns: list
    confidence: float = 1.0
    overlay: Optional[bytes] = None        # JPEG с рамками столбцов
    overlay_size: tuple = (0, 0)
    angle: float = 0.0                     # на сколько градусов выпрямили
    elapsed_ms: float = 0.0
    steps_ms: dict = field(default_factory=dict)

    @property
    def todo(self) -> str:
        return "\n".join(self.columns)

    @property
    def translit(self) -> str:
        return "\n".join(self.translit_columns)

    @property
    def low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


# ------------------------------------------------------------------ модель

def column_line(column: Image.Image, img_h: int = 64) -> np.ndarray:
    """Столбец -> горизонтальная строка высотой img_h, чернила = 1.
    То же преобразование, что при обучении: поворот, масштаб, ширина
    кратна 4 (сеть сжимает ширину вчетверо)."""
    im = column.convert("L").rotate(90, expand=True)
    w, h = im.size
    im = im.resize((max(8, round(w * img_h / h)), img_h), Image.BILINEAR)
    a = (255 - np.asarray(im, dtype=np.float32)) / 255.0
    pad = (-a.shape[1]) % 4
    return np.pad(a, ((0, 0), (0, pad))) if pad else a


class Model:
    def __init__(self, path: str, threads: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OcrUnavailable("не установлен onnxruntime") from exc
        if not Path(path).exists():
            raise OcrUnavailable(f"нет файла модели: {path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.chars = json.loads(meta["chars"])     # индекс k в выходе = chars[k - 1]; 0 = blank
        self.img_h = int(meta["img_h"])
        self.path = path

    def read(self, columns: list):
        """-> (строки, средняя уверенность по всем непустым шагам)."""
        lines = [column_line(c, self.img_h) for c in columns]
        order = np.argsort([a.shape[1] for a in lines])   # близкие по ширине — в один батч
        texts = [""] * len(lines)
        conf_sum, conf_n = 0.0, 0
        for k in range(0, len(order), MAX_BATCH):
            idx = order[k:k + MAX_BATCH]
            width = max(lines[i].shape[1] for i in idx)
            x = np.zeros((len(idx), 1, self.img_h, width), np.float32)
            for j, i in enumerate(idx):
                x[j, 0, :, :lines[i].shape[1]] = lines[i]
            logits = self.sess.run(None, {"image": x})[0]
            for j, i in enumerate(idx):
                steps = logits[j, :lines[i].shape[1] // 4]
                best = steps.argmax(-1)
                # CTC: схлопываем повторы, убираем blank
                keep = (best != 0) & np.concatenate([[True], best[1:] != best[:-1]])
                texts[i] = "".join(self.chars[c - 1] for c in best[keep])
                p = np.exp(steps - steps.max(-1, keepdims=True))
                p = (p / p.sum(-1, keepdims=True)).max(-1)[best != 0]
                conf_sum += float(p.sum()); conf_n += len(p)
        return texts, (conf_sum / conf_n if conf_n else 0.0)


_model: Optional[Model] = None
_model_lock = threading.Lock()


def model_path() -> str:
    return os.environ.get("OCR_MODEL_PATH") or str(_DEFAULT_MODEL)


def get_model() -> Model:
    """Загруженная модель (грузится при первом обращении)."""
    global _model
    with _model_lock:
        if _model is None:
            threads = int(os.environ.get("OCR_THREADS") or 0) or min(4, os.cpu_count() or 1)
            _model = Model(model_path(), threads)
        return _model


def _split_page():
    try:
        from .page_layout import split_page
    except ImportError as exc:                   # разметке нужен scipy
        raise OcrUnavailable(f"не хватает зависимости: {exc.name}") from exc
    return split_page


def warmup() -> dict:
    """Загрузить модель и прогнать пустой столбец: первый запрос не ждёт."""
    _split_page()
    m = get_model()
    m.read([Image.new("L", (64, 400), 255)])
    return {"model_path": m.path, "alphabet": len(m.chars)}


# ---------------------------------------------------------------- картинка

def open_image(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        # JPEG можно декодировать сразу уменьшенным — быстрее и меньше памяти
        im.draft("L", (MAX_SIDE, MAX_SIDE))
        im = ImageOps.exif_transpose(im)        # фото с телефона, присланное файлом, бывает повёрнуто через EXIF
        im = im.convert("L")
    except Exception as exc:
        raise OcrError("Не получилось открыть картинку. Пришлите фото или файл PNG/JPG.") from exc
    if max(im.size) > MAX_SIDE:
        k = MAX_SIDE / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    return im


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)   # Pillow ≥ 10.1: масштабируемый встроенный шрифт
    except TypeError:
        return ImageFont.load_default()


def draw_columns(gray: np.ndarray, boxes: list):
    """Выпрямленная страница + рамки столбцов с номерами -> (JPEG, размер)."""
    im = Image.fromarray(gray).convert("RGB")
    k = min(1.0, OVERLAY_SIDE / max(im.size))
    if k < 1.0:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    line = max(2, round(max(im.size) / 500))
    widths = [(x1 - x0) * k for x0, _, x1, _ in boxes]
    font = _font(max(12, min(40, round((np.median(widths) if widths else 30) * 0.45))))
    for n, (x0, y0, x1, y1) in enumerate(boxes, 1):
        color = _BOX_COLORS[(n - 1) % 2]
        box = [round(x0 * k), round(y0 * k), round(x1 * k), round(y1 * k)]
        box = [max(0, box[0]), max(0, box[1]), min(im.width - 1, box[2]), min(im.height - 1, box[3])]
        draw.rectangle(box, outline=color, width=line)
        label = str(n)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        cx = (box[0] + box[2]) // 2
        ty = box[1] - th - 2 * line if box[1] - th - 2 * line >= 0 else box[1] + line
        draw.rectangle([cx - tw // 2 - line, ty - line, cx + tw // 2 + line, ty + th + line], fill=color)
        draw.text((cx - tw // 2, ty), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue(), im.size


# --------------------------------------------------------------- основное

def recognize(data: bytes, overlay: bool = True, punctuation: str = DEFAULT_PUNCT) -> OcrResult:
    """Байты картинки -> OcrResult. Синхронно и не быстро (~1 с на страницу).

    punctuation — знаки препинания в ответе (PUNCT_* из punctuation.py): по
    умолчанию убираются и те, что модель прочитала на странице."""
    started = time.perf_counter()
    steps = {}
    # OcrUnavailable — до того, как тратить время на картинку
    split_page = _split_page()
    model = get_model()

    t0 = time.perf_counter()
    img = open_image(data)
    steps["картинка"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cols, dbg = split_page(img, debug=True)
    steps["разметка"] = (time.perf_counter() - t0) * 1000
    if not cols:
        raise OcrError(
            "Не нашёл на картинке вертикального текста. Пришлите фото, где "
            "столбцы тодо бичиг идут сверху вниз."
        )

    t0 = time.perf_counter()
    todo, confidence = model.read(cols)
    todo = [tidy(c) for c in todo]
    steps["модель"] = (time.perf_counter() - t0) * 1000
    if sum(ch not in _NOT_LETTERS for t in todo for ch in t) < MIN_LETTERS:
        raise OcrError(
            "Не разобрал на картинке тодо бичиг. Нужно фото, где столбцы идут "
            "сверху вниз, а буквы крупные и чёткие."
        )

    t0 = time.perf_counter()
    # Страница размечена целиком — столбцы идут одним текстом: бирга перед
    # первым, четыре точки после последнего. Число строк не меняется, иначе
    # сбилась бы нумерация рамок.
    todo = apply_punctuation(normalize_j("\n".join(todo)), punctuation).split("\n")
    translit = [to_translit(t) for t in todo]
    steps["транслитерация"] = (time.perf_counter() - t0) * 1000

    res = OcrResult(columns=todo, translit_columns=translit, confidence=confidence,
                    angle=dbg["angle"], steps_ms=steps)
    if overlay:
        t0 = time.perf_counter()
        res.overlay, res.overlay_size = draw_columns(dbg["gray"], dbg["boxes"])
        steps["рамки"] = (time.perf_counter() - t0) * 1000
    res.elapsed_ms = (time.perf_counter() - started) * 1000
    return res
