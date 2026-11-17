"""Разовое преобразование из худум в тодо бичик."""

import asyncio
from html import escape

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message

from core.hudum import get_converter

from .. import texts
from ..formatting import MESSAGE_LIMIT

router = Router(name="hudum")


def _from_hudum(text: str) -> str:
    return get_converter().convert(text, "todo")


@router.message(Command("from_hudum"))
async def cmd_from_hudum(message: Message, command: CommandObject) -> None:
    text = command.args
    if not text or not text.strip():
        await message.answer(texts.FROM_HUDUM_USAGE)
        return

    try:
        result = await asyncio.to_thread(_from_hudum, text)
    except ValueError:
        await message.answer(texts.FROM_HUDUM_USAGE)
        return
    reply = f"{texts.INLINE_TODO_TITLE}\n<code>{escape(result)}</code>"
    if len(reply) <= MESSAGE_LIMIT:
        await message.answer(reply)
    else:
        await message.answer_document(
            BufferedInputFile(result.encode("utf-8"), filename="todo.txt"),
            caption=texts.INLINE_TODO_TITLE,
        )
