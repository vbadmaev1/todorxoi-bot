# -*- coding: utf-8 -*-
"""
/settings — цвет текста, цвет фона и размер шрифта для картинки,
исправление текста, набранного без калмыцких букв (core/fix_letters.py;
по умолчанию выключено — правильно набранный текст не должен меняться
без спроса), и знаки препинания тодо бичиг (core/punctuation.py; по
умолчанию не ставятся — ни в тексте, ни на картинке, ни в распознанном
фото) и время работы под ответом (по умолчанию показывается).

Настройки у каждого пользователя свои и лежат в таблице user_settings,
то есть переживают перезапуск бота (в отличие от режима, который живёт в
памяти FSM). Настройки картинки на режимы /translit и /todo не влияют —
там текст.

Два поведения, ради которых всё и затевалось:

  • прозрачный фон — картинку с ним нельзя слать как фото: Telegram
    пережмёт PNG в JPEG, и прозрачность станет чёрным прямоугольником.
    Такие картинки уходят документом (см. handlers/translate.py);

  • совпавшие цвета текста и фона — рисовать нечитаемый прямоугольник
    бессмысленно, поэтому рендер откатывается на чёрное по белому и
    честно сообщает об этом под картинкой.
"""

import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from core import DEFAULT_PUNCT, FONT_SIZES, FONTS, PUNCT_MODES, ImageOptions
from core.todo_image import color_label

from .. import keyboards, texts
from ..storage import Storage

log = logging.getLogger(__name__)
router = Router(name="settings")

_VALID_COLORS = {key for key, _, _ in keyboards.BG_PALETTE}


async def load_options(storage: Storage, user_id: int) -> ImageOptions:
    """Настройки пользователя -> ImageOptions. Незаполненные поля берут
    значения по умолчанию, мусор из БД игнорируется."""
    saved = await storage.get_settings(user_id)
    opts = ImageOptions()
    if saved.get("fg") in _VALID_COLORS:
        opts.fg = saved["fg"]
    if saved.get("bg") in _VALID_COLORS:
        opts.bg = saved["bg"]
    if saved.get("size") in FONT_SIZES:
        opts.size = saved["size"]
    if saved.get("font") in FONTS:
        opts.font = saved["font"]
    return opts


async def load_fix_letters(storage: Storage, user_id: int) -> bool:
    """Исправлять ли текст без калмыцких букв. По умолчанию — нет."""
    return (await storage.get_settings(user_id)).get("fix_letters") == "on"


async def load_punctuation(storage: Storage, user_id: int) -> str:
    """Знаки препинания тодо бичиг: off | frame | all. По умолчанию — off."""
    saved = (await storage.get_settings(user_id)).get("punctuation")
    return saved if saved in PUNCT_MODES else DEFAULT_PUNCT


async def load_show_time(storage: Storage, user_id: int) -> bool:
    """Показывать ли время работы под ответом. По умолчанию — да."""
    return (await storage.get_settings(user_id)).get("show_time") != "off"


def _describe(opts: ImageOptions, fix_letters: bool, punct: str, show_time: bool) -> str:
    return texts.SETTINGS.format(
        fg=color_label(opts.fg),
        bg=color_label(opts.bg),
        size=keyboards.SIZE_LABELS.get(opts.size, opts.size),
        font=keyboards.FONT_LABELS.get(opts.font, opts.font),
        fix_letters=texts.FIX_LETTERS_ON if fix_letters else texts.FIX_LETTERS_OFF,
        punctuation=keyboards.PUNCT_LABELS.get(punct, punct),
        show_time=texts.SHOW_TIME_ON if show_time else texts.SHOW_TIME_OFF,
    )


@router.message(Command("settings"))
async def cmd_settings(message: Message, storage: Storage) -> None:
    user_id = message.from_user.id
    opts = await load_options(storage, user_id)
    fix = await load_fix_letters(storage, user_id)
    punct = await load_punctuation(storage, user_id)
    show_time = await load_show_time(storage, user_id)
    await message.answer(
        _describe(opts, fix, punct, show_time),
        reply_markup=keyboards.settings_menu(opts, fix, punct, show_time),
    )


async def _refresh(
    callback: CallbackQuery, opts: ImageOptions, fix: bool, punct: str,
    show_time: bool, markup=None,
) -> None:
    """Перерисовать экран настроек. Без markup — главное меню."""
    if markup is None:
        markup = keyboards.settings_menu(opts, fix, punct, show_time)
    try:
        await callback.message.edit_text(
            _describe(opts, fix, punct, show_time), reply_markup=markup
        )
    except TelegramBadRequest:
        # «message is not modified» — значит показывать уже нечего
        pass


@router.callback_query(F.data.startswith(f"{keyboards.CB_SETTINGS}:"))
async def on_settings(callback: CallbackQuery, storage: Storage) -> None:
    parts = callback.data.split(":")
    action = parts[1] if len(parts) > 1 else ""
    user_id = callback.from_user.id
    opts = await load_options(storage, user_id)
    fix = await load_fix_letters(storage, user_id)
    punct = await load_punctuation(storage, user_id)
    show_time = await load_show_time(storage, user_id)

    if action == "pick":
        field = parts[2]
        if field == "size":
            markup = keyboards.size_picker(opts.size)
        elif field == "font":
            markup = keyboards.font_picker(opts.font)
        elif field == "punctuation":
            markup = keyboards.punct_picker(punct)
        else:
            markup = keyboards.color_picker(field, getattr(opts, field))
        await callback.answer()
        await _refresh(callback, opts, fix, punct, show_time, markup)
        return

    if action == "toggle" and parts[2:3] == ["fix_letters"]:
        fix = not fix
        await storage.set_setting(user_id, "fix_letters", "on" if fix else "off")
        await callback.answer(texts.SETTINGS_FIX_ON if fix else texts.SETTINGS_FIX_OFF)
        await _refresh(callback, opts, fix, punct, show_time)
        return

    if action == "toggle" and parts[2:3] == ["show_time"]:
        show_time = not show_time
        await storage.set_setting(user_id, "show_time", "on" if show_time else "off")
        await callback.answer(
            texts.SETTINGS_TIME_ON if show_time else texts.SETTINGS_TIME_OFF
        )
        await _refresh(callback, opts, fix, punct, show_time)
        return

    if action == "set":
        field, value = parts[2], parts[3]
        ok = ((field in ("fg", "bg") and value in _VALID_COLORS)
              or (field == "size" and value in FONT_SIZES)
              or (field == "font" and value in FONTS)
              or (field == "punctuation" and value in PUNCT_MODES))
        if not ok:
            await callback.answer()
            return
        await storage.set_setting(user_id, field, value)
        opts = await load_options(storage, user_id)
        punct = await load_punctuation(storage, user_id)
        # предупреждаем сразу, а не когда человек получит чёрно-белую картинку
        if field in ("fg", "bg") and opts.fg == opts.bg:
            await callback.answer(texts.SETTINGS_SAME_COLOR, show_alert=True)
        else:
            await callback.answer(texts.SETTINGS_SAVED)
        await _refresh(callback, opts, fix, punct, show_time)
        return

    if action == "back":
        await callback.answer()
        await _refresh(callback, opts, fix, punct, show_time)
        return

    if action == "reset":
        await storage.reset_settings(user_id)
        await callback.answer(texts.SETTINGS_RESET)
        await _refresh(callback, ImageOptions(), False, DEFAULT_PUNCT, True)
        return

    await callback.answer()
