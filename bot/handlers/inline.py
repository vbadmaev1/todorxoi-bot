# -*- coding: utf-8 -*-
"""Исправление калмыцких букв через @имя_бота в любом чате."""

import asyncio

from aiogram import Router
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent

import core

from .. import texts

router = Router(name="inline_fix")


@router.inline_query()
async def fix_inline(query: InlineQuery) -> None:
    text = query.query.strip()
    results = []
    if text:
        try:
            result = await asyncio.to_thread(core.process, text, core.TARGET_FIX)
        except core.PipelineError:
            # Telegram запрашивает результаты и для неполного ввода.
            result = None
        if result and result.fixed_text:
            results.append(
                InlineQueryResultArticle(
                    id="fix",
                    title=result.fixed_text,
                    description=(
                        texts.INLINE_FIX_SEND
                        if result.fixed_text != text
                        else texts.INLINE_FIX_UNCHANGED
                    ),
                    input_message_content=InputTextMessageContent(
                        message_text=result.fixed_text,
                        parse_mode=None,
                    ),
                )
            )
    await query.answer(results, cache_time=0, is_personal=True)
