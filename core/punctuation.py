# -*- coding: utf-8 -*-
"""
punctuation.py — знаки препинания для картинки.

В тексте эти знаки не ставятся: текстовый ответ /todo — это просто запись
слов, а бирга и четыре точки размечают лист, то есть относятся именно к
изображению. Поэтому функция вызывается только на пути рендера.

Что делает:

  ᠀  бирга (U+1800)      — в самом начале текста, знак начала записи;
  ᠂  запятая (U+1802)    — вместо обычной;
  ᠃  точка (U+1803)      — конец предложения внутри текста;
  ᠅  четыре точки (U+1805) — конец всего текста.

Точка в самом конце заменяется на четыре точки. Если текст кончается
вопросительным или восклицательным знаком, знак остаётся, а четыре точки
добавляются после него.

На вход приходит уже готовый юникод тодо бичиг, где знаки препинания
записаны вертикальными презентационными формами (︒ ︐ ︕ ︖) — так их
превращает таблица в translit_todo.py.
"""

BIRGA = "᠀"
COMMA = "᠂"
FULL_STOP = "᠃"
FOUR_DOTS = "᠅"

# то, что приезжает из таблицы translit_todo.py
_SRC_COMMA = "︐"
_SRC_STOP = "︒"
_SRC_EXCL = "︕"
_SRC_QUES = "︖"

_TERMINAL = (_SRC_EXCL, _SRC_QUES)


def add_punctuation(todo_text: str, birga: bool = True) -> str:
    """Расставляет знаки препинания тодо бичиг в готовой строке."""
    if not todo_text:
        return todo_text

    text = todo_text.replace(_SRC_COMMA, COMMA)

    # хвост считаем по тексту без концевых пробелов и переводов строк,
    # чтобы «слово.\n» тоже посчиталось концом текста
    stripped = text.rstrip()
    trailing = text[len(stripped):]
    last = stripped[-1] if stripped else ""

    if last == _SRC_STOP:
        # последняя точка — это конец всего текста
        body = stripped[:-1].replace(_SRC_STOP, FULL_STOP)
        text = body + FOUR_DOTS + trailing
    elif last in _TERMINAL:
        # вопрос или восклицание оставляем и дописываем четыре точки
        body = stripped[:-1].replace(_SRC_STOP, FULL_STOP)
        text = body + last + FOUR_DOTS + trailing
    else:
        # текст без концевого знака — четыре точки не навязываем
        text = text.replace(_SRC_STOP, FULL_STOP)

    return (BIRGA + text) if birga else text
