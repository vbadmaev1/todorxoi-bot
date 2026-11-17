# -*- coding: utf-8 -*-
"""Исправление калмыцкого текста и запись тодо бичик через @имя_бота."""

import asyncio
import logging

from aiogram import Router
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent

import core

from .. import texts

log = logging.getLogger(__name__)
router = Router(name="inline_fix")


def _inline_results(text: str) -> list[InlineQueryResultArticle]:
    try:
        fixed = core.process(text, core.TARGET_FIX).fixed_text
    except core.PipelineError:
        # Telegram запрашивает результаты и для неполного ввода.
        return []
    if not fixed:
        return []

    results = [
        InlineQueryResultArticle(
            id="fix",
            title=fixed,
            description=texts.INLINE_FIX_SEND if fixed != text else texts.INLINE_FIX_UNCHANGED,
            input_message_content=InputTextMessageContent(message_text=fixed, parse_mode=None),
        )
    ]
    try:
        todo = core.process(fixed, core.TARGET_TODO).todo
    except Exception:
        log.exception("не удалось записать inline-текст тодо бичик")
        return results
    if todo:
        results.append(
            InlineQueryResultArticle(
                id="todo",
                title=texts.INLINE_TODO_TITLE,
                description=todo,
                input_message_content=InputTextMessageContent(message_text=todo, parse_mode=None),
            )
        )
    return results


@router.inline_query()
async def fix_inline(query: InlineQuery) -> None:
    text = query.query.strip()
    results = await asyncio.to_thread(_inline_results, text) if text else []
    await query.answer(results, cache_time=0, is_personal=True)
