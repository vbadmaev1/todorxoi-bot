# -*- coding: utf-8 -*-
"""Сборка текста ответа: результат + «вход распознан как…» + время работы."""

from html import escape

from core import Result

from . import texts

TARGET_ICONS = {"translit": "🔤", "todo": "ᡐ", "image": "🖼"}

CAPTION_LIMIT = 1024
MESSAGE_LIMIT = 4096


def fmt_ms(ms: float) -> str:
    """412.7 -> «413 мс», 1234.5 -> «1,23 с» — читается легче, чем голые ms."""
    if ms < 1000:
        return f"{ms:.0f} мс"
    return f"{ms / 1000:.2f} с".replace(".", ",")


def timing_line(res: Result) -> str:
    """Только общее время. Разбивка по шагам и статистика «сколько слов из
    словаря» — внутренняя кухня: она есть в БД для анализа, но человеку в
    чате не нужна."""
    return f"⏱ {fmt_ms(res.elapsed_ms)}"


def _header(res: Result) -> str:
    icon = TARGET_ICONS.get(res.target, "•")
    title = texts.MODE_TITLES.get(res.target, res.target).split(" ", 1)[-1]
    return f"{icon} <b>{title}</b>"


def render_result(res: Result) -> str:
    """Текст ответа для режимов «транслитерация» и «тодо бичиг»."""
    blocks = [_header(res)]

    if res.target == "translit":
        blocks.append(f"<code>{escape(res.translit or '')}</code>")
    else:
        if res.translit:
            blocks.append(
                f"Транслитерация:\n<code>{escape(res.translit)}</code>"
            )
        blocks.append(f"Тодо бичиг:\n<code>{escape(res.todo or '')}</code>")

    blocks.append(timing_line(res))
    return "\n\n".join(blocks)


def render_caption(res: Result, page: int = 1, pages: int = 1) -> str:
    """Подпись под картинкой — то же самое, но с оглядкой на лимит в 1024."""
    text = (
        render_result(res)
        if res.target != "image"
        else _image_caption(res, page, pages)
    )
    if len(text) <= CAPTION_LIMIT:
        return text
    return text[: CAPTION_LIMIT - 1] + "…"


def _image_caption(res: Result, page: int = 1, pages: int = 1) -> str:
    # под картинкой транслитерацию не дублируем: кому она нужна, тот
    # спросит её отдельно командой /translit
    header = _header(res)
    if pages > 1:
        # Столбцы читаются слева направо, листы — по порядку. Номер нужен
        # именно в подписи: в чате сообщения легко перепутать местами.
        header += f" · картинка {page} из {pages}"
    blocks = [header]

    # служебное говорим один раз, под первым листом: под каждым — шум
    if page == 1:
        if not res.shaping_ok:
            from core.todo_image import SHAPING_WARNING

            blocks.append(SHAPING_WARNING)
        if res.color_fallback:
            blocks.append(texts.COLOR_FALLBACK_NOTE)
        if res.pages_dropped:
            blocks.append(texts.PAGES_DROPPED.format(n=res.pages_dropped))

    if page == pages:
        blocks.append(timing_line(res))
    return "\n\n".join(blocks)


def render_ocr(res) -> list:
    """Ответ на фото: транслитерация и тодо бичиг, по строке на столбец.

    Полная страница — это около 2000 символов на обе записи, почти всегда
    одно сообщение. Если не влезает в лимит Telegram, режем по столбцам:
    каждая часть — законченный блок <code>, чтобы копировать было удобно.
    Возвращает список текстов сообщений."""
    blocks = [f"📷 <b>Текст с фото</b> · столбцов: {len(res.columns)}"]
    blocks += _code_blocks("Транслитерация:", res.translit_columns)
    blocks += _code_blocks("Тодо бичиг:", res.columns)
    if res.low_confidence:
        blocks.append(texts.OCR_LOW_CONFIDENCE)
    blocks.append(f"⏱ {fmt_ms(res.elapsed_ms)}")
    return _pack(blocks, MESSAGE_LIMIT)


def _code_blocks(title: str, lines: list, limit: int = 3500) -> list:
    """Строки -> блоки «заголовок + <code>…</code>» не длиннее limit."""
    chunks, cur = [], []
    for line in lines:
        if cur and len(escape("\n".join(cur + [line]))) > limit:
            chunks.append(cur)
            cur = []
        cur.append(line)
    chunks.append(cur)
    blocks = []
    for k, chunk in enumerate(chunks):
        body = escape("\n".join(chunk))
        blocks.append((title + "\n" if k == 0 else "") + f"<code>{body}</code>")
    return blocks


def _pack(blocks: list, limit: int) -> list:
    """Блоки -> сообщения: подряд, пока влезает в limit."""
    messages = []
    for block in blocks:
        if messages and len(messages[-1]) + 2 + len(block) <= limit:
            messages[-1] += "\n\n" + block
        else:
            messages.append(block)
    return messages
