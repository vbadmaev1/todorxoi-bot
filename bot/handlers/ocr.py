# -*- coding: utf-8 -*-
"""Фото → текст: распознавание тодо бичиг (core/ocr.py).

Фото принимается в любом режиме: режимы касаются того, во что превращать
текст, а картинку ни с чем не спутать. Команда /ocr — подсказка; с фото
(в подписи) или ответом на сообщение с фото — распознаёт его. Картинка,
присланная файлом, тоже годится — её Telegram не пережимает.

Ответ: картинка с рамками и номерами столбцов (как бот разрезал страницу),
затем кириллица, транслитерация и юникод тодо бичиг по строке на столбец, под ними
👍/👎. Если всё влезает в подпись — одним сообщением.

Распознавание — около секунды счётной работы и ~160 МБ памяти на страницу:
  • уезжает в поток (asyncio.to_thread), чтобы бот не вставал для всех;
  • идёт по одной картинке за раз (_lock): две страницы параллельно удвоили
    бы пик памяти, а быстрее на сервере с одним-двумя ядрами не стало бы.
"""

import asyncio
import logging
from typing import Optional, Tuple

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, Message

from core import ocr as core_ocr

from .. import formatting, keyboards, texts
from ..config import Config
from ..storage import Storage
from .settings import load_punctuation, load_show_time
from .translate import _PHOTO_MAX_RATIO, _send_patiently

log = logging.getLogger(__name__)
router = Router(name="ocr")

TARGET_OCR = "ocr"
# больше этого Bot API боту файл не отдаёт
MAX_FILE_BYTES = 20 * 1024 * 1024

_lock = asyncio.Lock()


def _image_file(message: Optional[Message]) -> Optional[Tuple[str, Optional[int]]]:
    """(file_id, размер) картинки из сообщения или None."""
    if message is None:
        return None
    if message.photo:
        size = message.photo[-1]                    # самый крупный вариант
        return size.file_id, size.file_size
    doc = message.document
    if doc and (doc.mime_type or "").lower().startswith("image/"):
        return doc.file_id, doc.file_size
    return None


@router.message(Command("ocr"))
async def cmd_ocr(
    message: Message, state: FSMContext, storage: Storage, config: Config
) -> None:
    # как и переключение режима, команда отменяет незаконченный ввод исправления
    await state.set_state(None)
    picked = _image_file(message) or _image_file(message.reply_to_message)
    if picked:
        await read_image(message, *picked, storage=storage, config=config)
    else:
        await message.answer(texts.OCR_HINT)


@router.message(StateFilter(None), F.photo | F.document)
async def on_image(message: Message, storage: Storage, config: Config) -> None:
    picked = _image_file(message)
    if picked:
        await read_image(message, *picked, storage=storage, config=config)
        return
    doc = message.document
    is_pdf = (doc.mime_type or "").lower() == "application/pdf" or (
        doc.file_name or ""
    ).lower().endswith(".pdf")
    await message.answer(texts.OCR_PDF if is_pdf else texts.OCR_NOT_IMAGE)


async def read_image(
    message: Message,
    file_id: str,
    file_size: Optional[int],
    storage: Storage,
    config: Config,
) -> None:
    if file_size and file_size > MAX_FILE_BYTES:
        await message.answer(texts.OCR_TOO_BIG)
        return

    user = message.from_user
    base = {
        "user_id": user.id if user else None,
        "username": user.username if user else None,
        "chat_id": message.chat.id,
        "target": TARGET_OCR,
        "source_script": "photo",
        # саму картинку не храним: по file_id её можно заново скачать через
        # Bot API, если исправление понадобится для дообучения
        "input_text": f"photo:{file_id}",
    }

    punctuation = (
        await load_punctuation(storage, user.id) if user else core_ocr.DEFAULT_PUNCT
    )
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        data = (await message.bot.download(file_id)).getvalue()
        async with _lock:
            res = await asyncio.to_thread(
                core_ocr.recognize, data, config.ocr_overlay, punctuation
            )
    except core_ocr.OcrUnavailable as exc:
        log.warning("распознавание фото недоступно: %s", exc)
        await storage.save_request(ok=False, error=f"ocr unavailable: {exc}", **base)
        await message.answer(texts.OCR_UNAVAILABLE)
        return
    except core_ocr.OcrError as exc:
        await storage.save_request(ok=False, error=str(exc), **base)
        await message.answer(f"⚠️ {exc}")
        return
    except Exception:
        log.exception("сбой распознавания фото: file_id=%s", file_id)
        await storage.save_request(ok=False, error="internal", **base)
        await message.answer(texts.ERROR_GENERIC)
        return

    request_id = await storage.save_request(
        translit=res.translit,
        todo=res.todo,
        cyrillic=res.cyrillic or None,
        elapsed_ms=res.elapsed_ms,
        steps=res.steps_ms,
        ok=True,
        **base,
    )
    markup = keyboards.result_keyboard(request_id, TARGET_OCR)
    show_time = await load_show_time(storage, user.id) if user else True
    await _answer(message, res, markup, show_time)


async def _answer(message: Message, res, markup, show_time: bool = True) -> None:
    parts = formatting.render_ocr(res, show_time)
    if res.overlay:
        photo = BufferedInputFile(res.overlay, filename="columns.jpg")
        w, h = res.overlay_size
        # один столбец — картинка узкая и высокая; такую Telegram как фото не примет
        send = (
            message.answer_document
            if max(w, h) / max(1, min(w, h)) > _PHOTO_MAX_RATIO
            else message.answer_photo
        )
        if len(parts) == 1 and len(parts[0]) <= formatting.CAPTION_LIMIT:
            await _send_patiently(send, photo, caption=parts[0], reply_markup=markup)
            return
        await _send_patiently(send, photo, caption=texts.OCR_OVERLAY_CAPTION)
    for k, part in enumerate(parts):
        last = k == len(parts) - 1
        await _send_patiently(message.answer, part, reply_markup=markup if last else None)
