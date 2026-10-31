# -*- coding: utf-8 -*-
"""/jangar — мини-приложение «Джангар» (Telegram Mini App): эпос параллельно — скан издания Н. Очирова на
тодо бичиг (1910), кириллица и перевод С. Липкина; слово — тодо, латиница и словарь Б. Х. Тодаевой.

Само приложение — статическая страница (research/jangar/web), её адрес — JANGAR_URL в .env (только HTTPS:
Telegram открывает мини-приложения лишь по https). Бот присылает кнопку, которая открывает страницу отдельным
окном Telegram. «/jangar 5» — сразу пятая песнь (номера глав halmg.ru: 0 — вступление, 1–12 — песни).
"""

from typing import Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, WebAppInfo

from .. import texts
from ..config import Config

router = Router(name="jangar")


def jangar_url(base: str, chapter: Optional[int] = None) -> str:
    """Адрес приложения; глава — параметром ?ch= (хэш Telegram занимает своими параметрами запуска)."""
    if chapter is None:
        return base
    parts = urlsplit(base)
    query = (parts.query + "&" if parts.query else "") + urlencode({"ch": chapter})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def jangar_keyboard(url: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=texts.BTN_JANGAR, web_app=WebAppInfo(url=url))]]
    )


@router.message(Command("jangar"))
async def cmd_jangar(message: Message, command: CommandObject, config: Config) -> None:
    if not config.jangar_url:
        await message.answer(texts.JANGAR_OFF)
        return
    chapter = None
    arg = (command.args or "").strip()
    if arg.isdigit() and 0 <= int(arg) <= 12:
        chapter = int(arg)
    await message.answer(texts.JANGAR, reply_markup=jangar_keyboard(jangar_url(config.jangar_url, chapter)))
