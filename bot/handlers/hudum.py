"""One-shot conversion to Hudum and back to Todo in Telegram."""

import asyncio
from html import escape

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message

import core
from core.hudum import get_converter
from core.translit_todo import SCRIPT_TODO

from .. import texts
from ..formatting import MESSAGE_LIMIT
from ..storage import Storage
from .settings import load_fix_letters, load_punctuation

router = Router(name="hudum")


def _convert(text: str, target: str, fix_letters: bool, punctuation: str) -> str:
    if target == "hudum" and core.detect_script(text) != SCRIPT_TODO:
        converted = core.process(
            text, core.TARGET_TODO, fix_letters=fix_letters, punctuation=punctuation
        )
        text = converted.todo or ""
    return get_converter().convert(text, target)


@router.message(Command("hudum", "from_hudum"))
async def cmd_hudum(message: Message, command: CommandObject, storage: Storage) -> None:
    target = "hudum" if command.command == "hudum" else "todo"
    text = command.args
    if not text or not text.strip():
        await message.answer(texts.HUDUM_USAGE[target])
        return

    fix_letters = False
    punctuation = core.DEFAULT_PUNCT
    if target == "hudum" and message.from_user and core.detect_script(text) != SCRIPT_TODO:
        fix_letters = await load_fix_letters(storage, message.from_user.id)
        punctuation = await load_punctuation(storage, message.from_user.id)

    try:
        result = await asyncio.to_thread(_convert, text, target, fix_letters, punctuation)
    except core.PipelineError as error:
        await message.answer(escape(str(error)))
        return
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
