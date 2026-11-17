"""One-shot conversion between Todo and Hudum in Telegram."""

import asyncio
from html import escape

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message

from core.hudum import get_converter

from .. import texts
from ..formatting import MESSAGE_LIMIT

router = Router(name="hudum")


def _convert(text: str, target: str) -> str:
    return get_converter().convert(text, target)


@router.message(Command("hudum", "from_hudum"))
async def cmd_hudum(message: Message, command: CommandObject) -> None:
    target = "hudum" if command.command == "hudum" else "todo"
    text = command.args
    if not text or not text.strip():
        await message.answer(texts.HUDUM_USAGE[target])
        return

    try:
        result = await asyncio.to_thread(_convert, text, target)
    except ValueError:
        await message.answer(texts.HUDUM_USAGE[target])
        return
    title = texts.HUDUM_RESULT[target]
    reply = f"{title}\n<code>{escape(result)}</code>"
    if len(reply) <= MESSAGE_LIMIT:
        await message.answer(reply)
    else:
        await message.answer_document(
            BufferedInputFile(result.encode("utf-8"), filename=f"{target}.txt"), caption=title
        )
