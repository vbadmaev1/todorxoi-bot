# -*- coding: utf-8 -*-
"""Сборка текста ответа: результат + «вход распознан как…» + время работы."""

from html import escape

from core import Result

from . import texts

TARGET_ICONS = {"translit": "🔤", "todo": "ᡐ", "image": "🖼", "fix": "✏️"}

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


def render_result(res: Result, show_time: bool = True) -> str:
    """Текст ответа для режимов «транслитерация» и «тодо бичиг»."""
    blocks = [_header(res)]

    if res.target == "fix":
        return _render_fix(res, show_time)
    if res.target == "translit" and res.cyrillic:
        # тодо бичиг или латиница -> кириллица; транслитерацию латиницы
        # повторять незачем
        if res.source_script != "translit":
            blocks.append(f"{texts.LABEL_TRANSLIT}\n<code>{escape(res.translit or '')}</code>")
        blocks.append(f"{texts.LABEL_CYRILLIC}\n<code>{escape(res.cyrillic)}</code>")
    elif res.target == "translit":
        blocks.append(f"<code>{escape(res.translit or '')}</code>")
    else:
        if res.translit:
            blocks.append(
                f"{texts.LABEL_TRANSLIT}\n<code>{escape(res.translit)}</code>"
            )
        blocks.append(f"{texts.LABEL_TODO}\n<code>{escape(res.todo or '')}</code>")

    note = letters_note(res)
    if note:
        blocks.append(note)
    if show_time:
        blocks.append(timing_line(res))
    return "\n\n".join(blocks)


def _render_fix(res: Result, show_time: bool = True) -> str:
    """Режим /fix: исправленный текст целиком (его удобно скопировать) и
    что именно поменялось."""
    blocks = [_header(res)]
    if res.letter_fixes:
        blocks.append(f"{texts.FIX_TITLE}\n<code>{escape(res.fixed_text or '')}</code>")
        blocks.append(letters_note(res))
    else:
        blocks.append(texts.FIX_NOTHING)
    if show_time:
        blocks.append(timing_line(res))
    return "\n\n".join(blocks)


# сколько исправленных слов показывать, остальные — «и ещё N»
LETTER_FIXES_SHOWN = 12


def letters_note(res: Result) -> str:
    """Что сделано с калмыцкими буквами: список исправлений, если
    исправление включено, или подсказка про /settings, если выключено, а
    текст похож на набранный без ә ө ү һ җ ң."""
    if res.letter_fixes:
        shown = res.letter_fixes[:LETTER_FIXES_SHOWN]
        pairs = ", ".join(f"{escape(a)} → {escape(b)}" for a, b in shown)
        text = texts.LETTERS_FIXED.format(pairs=pairs)
        if len(res.letter_fixes) > len(shown):
            text += texts.LETTERS_FIXED_MORE.format(n=len(res.letter_fixes) - len(shown))
        return text
    if res.letters_flag:
        return texts.LETTERS_HINT
    return ""


def render_caption(
    res: Result, page: int = 1, pages: int = 1, show_time: bool = True
) -> str:
    """Подпись под картинкой — то же самое, но с оглядкой на лимит в 1024."""
    text = (
        render_result(res, show_time)
        if res.target != "image"
        else _image_caption(res, page, pages, show_time)
    )
    if len(text) <= CAPTION_LIMIT:
        return text
    return text[: CAPTION_LIMIT - 1] + "…"


def _image_caption(
    res: Result, page: int = 1, pages: int = 1, show_time: bool = True
) -> str:
    # под картинкой транслитерацию не дублируем: кому она нужна, тот
    # спросит её отдельно командой /translit
    header = _header(res)
    if pages > 1:
        # Столбцы читаются слева направо, листы — по порядку. Номер нужен
        # именно в подписи: в чате сообщения легко перепутать местами.
        header += texts.IMAGE_PAGE.format(page=page, pages=pages)
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
        note = letters_note(res)
        if note:
            blocks.append(note)

    if page == pages and show_time:
        blocks.append(timing_line(res))
    return "\n\n".join(blocks)


def columns_word(n: int) -> str:
    """3 -> «3 столбца»: 1 столбец, 2–4 столбца, 5–20 столбцов, 21 столбец..."""
    if n % 10 == 1 and n % 100 != 11:
        word = "столбец"
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        word = "столбца"
    else:
        word = "столбцов"
    return f"{n} {word}"


def render_ocr(res, show_time: bool = True) -> list:
    """Ответ на фото: кириллица, транслитерация и тодо бичиг, по строке на столбец.

    Полная страница — это около 2000 символов на обе записи, почти всегда
    одно сообщение. Если не влезает в лимит Telegram, режем по столбцам:
    каждая часть — законченный блок <code>, чтобы копировать было удобно.
    Возвращает список текстов сообщений."""
    blocks = [texts.OCR_TITLE.format(n=len(res.columns))]
    if res.cyrillic_columns:
        blocks += _code_blocks(texts.LABEL_CYRILLIC, res.cyrillic_columns)
    blocks += _code_blocks(texts.LABEL_TRANSLIT, res.translit_columns)
    blocks += _code_blocks(texts.LABEL_TODO, res.columns)
    if res.unreadable:
        blocks.append(
            texts.OCR_UNREADABLE.format(cols=columns_word(res.unreadable))
            + (texts.OCR_UNREADABLE_OVERLAY if res.overlay else "")
            + texts.OCR_UNREADABLE_HINT
        )
    if res.low_confidence:
        blocks.append(texts.OCR_LOW_CONFIDENCE)
    if show_time:
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
