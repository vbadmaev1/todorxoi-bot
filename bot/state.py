# -*- coding: utf-8 -*-
"""Состояние обратной связи и режим пользователя в чате."""

from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from .storage import Storage

DEFAULT_MODE = "translit"
MODES = ("translit", "todo", "hudum", "image", "fix")


class Feedback(StatesGroup):
    # бот ждёт от пользователя правильный вариант после нажатия 👎
    waiting_correction = State()


async def get_mode(state: FSMContext, storage: Storage) -> str:
    mode = await storage.get_mode(state.key.chat_id, state.key.user_id)
    return mode if mode in MODES else DEFAULT_MODE


async def set_mode(state: FSMContext, storage: Storage, mode: str) -> None:
    await storage.set_mode(state.key.chat_id, state.key.user_id, mode)
