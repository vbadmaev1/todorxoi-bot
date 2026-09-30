# -*- coding: utf-8 -*-
"""Сборка роутеров.

Порядок подключения важен, aiogram проверяет роутеры сверху вниз:

  1. common   — команды и кнопки меню. Без фильтра по состоянию, поэтому
                работают даже когда бот ждёт текст исправления.
  2. ocr      — /ocr (тоже в любом состоянии) и фото/картинки файлом. Раньше
                translate: там общий ответ «я понимаю только текст».
  3. feedback — 👍/👎 и приём исправления (ловит текст в своём состоянии).
  4. settings — /settings и кнопки палитры.
  5. admin    — /stats и /export: должны сработать раньше, чем общий
                обработчик текста примет команду за калмыцкое слово.
  6. translate — всё остальное: любой текст в текущем режиме.

Плюс общий обработчик ошибок: что бы ни упало внутри хэндлера (рендер,
отправка в Telegram), человек получает ответ, а не тишину.
"""

import logging

from aiogram import Router
from aiogram.types import ErrorEvent

from .. import texts
from . import admin, common, feedback, ocr, settings, translate

log = logging.getLogger(__name__)


async def on_error(event: ErrorEvent) -> None:
    update = event.update
    log.error("необработанная ошибка в апдейте %s", update.update_id,
              exc_info=event.exception)
    if update.callback_query:
        message = update.callback_query.message
    else:
        message = update.message
    if message is None:
        return
    try:
        await message.answer(texts.ERROR_GENERIC)
    except Exception:
        # не смогли даже ответить (бот заблокирован, сеть) — хватит лога
        log.warning("не удалось сообщить об ошибке в чат %s", message.chat.id)


def build_router() -> Router:
    root = Router(name="root")
    root.errors.register(on_error)
    root.include_router(common.router)
    root.include_router(ocr.router)
    root.include_router(feedback.router)
    root.include_router(settings.router)
    root.include_router(admin.router)
    root.include_router(translate.router)
    return root


__all__ = ["build_router"]
