# -*- coding: utf-8 -*-
"""Клавиатуры: постоянное меню режимов внизу и inline-кнопки под ответом."""

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from core.todo_image import BG_PALETTE, PALETTE, color_label  # noqa: F401

from . import texts

# callback_data:
#   fb:up:<request_id>      — палец вверх
#   fb:down:<request_id>    — палец вниз (дальше бот просит правильный ответ)
#   conv:todo:<request_id>  — «а теперь то же самое в тодо бичиг»
#   conv:image:<request_id>
#   set:pick:<поле>         — открыть палитру/список размеров
#   set:set:<поле>:<знач>   — выбрать значение
#   set:toggle:fix_letters  — исправлять текст без калмыцких букв: вкл/выкл
#   set:toggle:show_time    — время работы под ответом: вкл/выкл
#   set:set:punctuation:<режим> — знаки препинания: off | frame | all
#   set:back:-              — вернуться в меню настроек
#   set:reset:-             — сбросить всё на умолчания

CB_FEEDBACK = "fb"
CB_CONVERT = "conv"
CB_SETTINGS = "set"

# подписи живут в texts.py; здесь — для старых импортов keyboards.*_LABELS
SIZE_LABELS = texts.SIZE_LABELS
FONT_LABELS = texts.FONT_LABELS
PUNCT_LABELS = texts.PUNCT_LABELS


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=texts.BTN_TRANSLIT), KeyboardButton(text=texts.BTN_TODO)],
            [KeyboardButton(text=texts.BTN_IMAGE), KeyboardButton(text=texts.BTN_FIX)],
            [KeyboardButton(text=texts.BTN_HELP)],
        ],
        resize_keyboard=True,
        input_field_placeholder=texts.MENU_PLACEHOLDER,
    )


def result_keyboard(request_id: int, target: str, with_feedback: bool = True):
    """Кнопки под результатом: продолжить конвертацию + оценка."""
    rows = []

    convert = []
    if target == "fix":
        # исправленный текст — сразу дальше, в любой из трёх записей
        convert.append(
            InlineKeyboardButton(
                text=texts.BTN_TO_TRANSLIT, callback_data=f"{CB_CONVERT}:translit:{request_id}"
            )
        )
        convert.append(
            InlineKeyboardButton(
                text=texts.BTN_TO_TODO, callback_data=f"{CB_CONVERT}:todo:{request_id}"
            )
        )
    if target == "translit":
        convert.append(
            InlineKeyboardButton(
                text=texts.BTN_TO_TODO, callback_data=f"{CB_CONVERT}:todo:{request_id}"
            )
        )
    if target in ("translit", "todo", "fix"):
        convert.append(
            InlineKeyboardButton(
                text=texts.BTN_TO_IMAGE, callback_data=f"{CB_CONVERT}:image:{request_id}"
            )
        )
    if convert:
        rows.append(convert)

    if with_feedback:
        rows.append(
            [
                InlineKeyboardButton(
                    text="👍", callback_data=f"{CB_FEEDBACK}:up:{request_id}"
                ),
                InlineKeyboardButton(
                    text="👎", callback_data=f"{CB_FEEDBACK}:down:{request_id}"
                ),
            ]
        )

    if not rows:
        return None
    return InlineKeyboardMarkup(inline_keyboard=rows)


def settings_menu(
    opts, fix_letters: bool = False, punctuation: str = "off", show_time: bool = True
) -> InlineKeyboardMarkup:
    """Главный экран /settings: что менять."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_FG.format(value=color_label(opts.fg)),
                    callback_data=f"{CB_SETTINGS}:pick:fg",
                )
            ],
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_BG.format(value=color_label(opts.bg)),
                    callback_data=f"{CB_SETTINGS}:pick:bg",
                )
            ],
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_SIZE.format(
                        value=SIZE_LABELS.get(opts.size, opts.size)
                    ),
                    callback_data=f"{CB_SETTINGS}:pick:size",
                )
            ],
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_FONT.format(
                        value=FONT_LABELS.get(opts.font, opts.font)
                    ),
                    callback_data=f"{CB_SETTINGS}:pick:font",
                )
            ],
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_PUNCT.format(
                        value=PUNCT_LABELS.get(punctuation, punctuation)
                    ),
                    callback_data=f"{CB_SETTINGS}:pick:punctuation",
                )
            ],
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_FIX.format(
                        value=texts.FIX_LETTERS_ON if fix_letters else texts.FIX_LETTERS_OFF
                    ),
                    callback_data=f"{CB_SETTINGS}:toggle:fix_letters",
                )
            ],
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_TIME.format(
                        value=texts.SHOW_TIME_ON if show_time else texts.SHOW_TIME_OFF
                    ),
                    callback_data=f"{CB_SETTINGS}:toggle:show_time",
                )
            ],
            [
                InlineKeyboardButton(
                    text=texts.SET_BTN_RESET, callback_data=f"{CB_SETTINGS}:reset:-"
                )
            ],
        ]
    )


def _grid(buttons, per_row=3):
    return [buttons[i : i + per_row] for i in range(0, len(buttons), per_row)]


def color_picker(field: str, current: str) -> InlineKeyboardMarkup:
    """Палитра. Текущий цвет помечен галочкой, чтобы не гадать."""
    palette = BG_PALETTE if field == "bg" else PALETTE
    buttons = [
        InlineKeyboardButton(
            text=f"{chip} {label}" + (" ✓" if key == current else ""),
            callback_data=f"{CB_SETTINGS}:set:{field}:{key}",
        )
        for key, label, chip in palette
    ]
    rows = _grid(buttons, per_row=2)
    rows.append(
        [InlineKeyboardButton(text=texts.BTN_BACK, callback_data=f"{CB_SETTINGS}:back:-")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def size_picker(current: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=label + (" ✓" if key == current else ""),
            callback_data=f"{CB_SETTINGS}:set:size:{key}",
        )
        for key, label in SIZE_LABELS.items()
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            buttons,
            [InlineKeyboardButton(text=texts.BTN_BACK, callback_data=f"{CB_SETTINGS}:back:-")],
        ]
    )


def font_picker(current: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=label + (" ✓" if key == current else ""),
            callback_data=f"{CB_SETTINGS}:set:font:{key}",
        )
        for key, label in FONT_LABELS.items()
    ]
    rows = _grid(buttons, per_row=2)
    rows.append(
        [InlineKeyboardButton(text=texts.BTN_BACK, callback_data=f"{CB_SETTINGS}:back:-")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def punct_picker(current: str) -> InlineKeyboardMarkup:
    """Три режима знаков препинания — каждый своей строкой: подписи длинные."""
    rows = [
        [
            InlineKeyboardButton(
                text=label + (" ✓" if key == current else ""),
                callback_data=f"{CB_SETTINGS}:set:punctuation:{key}",
            )
        ]
        for key, label in PUNCT_LABELS.items()
    ]
    rows.append(
        [InlineKeyboardButton(text=texts.BTN_BACK, callback_data=f"{CB_SETTINGS}:back:-")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def rated_keyboard(rating: str) -> InlineKeyboardMarkup:
    """Клавиатура после оценки — кнопки убираем, оставляем отметку."""
    mark = texts.RATED_UP if rating == "up" else texts.RATED_DOWN
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=mark, callback_data="noop")]]
    )
