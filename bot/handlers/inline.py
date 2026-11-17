# -*- coding: utf-8 -*-
"""Исправление кириллицы и конвертация письма через @имя_бота."""

import asyncio
import logging

from aiogram import Router
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent

import core
from core.translit_todo import SCRIPT_CYRILLIC, SCRIPT_TODO

from .. import texts
from ..formatting import MESSAGE_LIMIT

log = logging.getLogger(__name__)
router = Router(name="inline_fix")


def _article(result_id: str, title: str, description: str, text: str) -> InlineQueryResultArticle:
    return InlineQueryResultArticle(
        id=result_id,
        title=title,
        description=description,
        input_message_content=InputTextMessageContent(message_text=text, parse_mode=None),
    )


def _inline_results(text: str) -> list[InlineQueryResultArticle]:
    script = core.detect_script(text)
    results = []
    if script == SCRIPT_CYRILLIC:
        try:
            fixed = core.process(text, core.TARGET_FIX).fixed_text
        except core.PipelineError:
            return []
        if not fixed:
            return []
        description = texts.INLINE_FIX_SEND if fixed != text else texts.INLINE_FIX_UNCHANGED
        results.append(_article("fix", fixed, description, fixed))
        text = fixed

    if script == SCRIPT_TODO:
        todo = text
    else:
        try:
            todo = core.process(text, core.TARGET_TODO).todo
        except core.PipelineError:
            return results
        except Exception:
            log.exception("не удалось записать inline-текст тодо бичик")
            return results
        if todo and len(todo) <= MESSAGE_LIMIT:
            results.append(_article("todo", texts.INLINE_TODO_TITLE, todo, todo))

    if todo:
        try:
            hudum = core.process(todo, core.TARGET_HUDUM).hudum
        except core.PipelineError:
            return results
        except Exception:
            log.exception("не удалось записать inline-текст худум бичик")
            return results
        if hudum and len(hudum) <= MESSAGE_LIMIT:
            results.append(_article("hudum", texts.HUDUM_TITLE, hudum, hudum))
    return results


@router.inline_query()
async def fix_inline(query: InlineQuery) -> None:
    text = query.query.strip()
    results = await asyncio.to_thread(_inline_results, text) if text else []
    await query.answer(results, cache_time=0, is_personal=True)
