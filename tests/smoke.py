# -*- coding: utf-8 -*-
"""
Прогон бота «на сухую», без подключения к Telegram: подменяем сессию
Bot API заглушкой и скармливаем диспетчеру настоящие Update. Проверяем,
что хэндлеры вообще подхватываются, порядок роутеров не перепутан и
кнопки/состояния отрабатывают.

Запуск (токен не нужен, сеть тоже):

    python -m tests.smoke

Никаких внешних зависимостей, кроме тех, что и так нужны боту.
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BOT_TOKEN", "123456:TEST")
os.environ.setdefault("ADMIN_IDS", "5")
os.environ.setdefault("WARMUP", "false")

from aiogram import Bot, Dispatcher  # noqa: E402
from aiogram.client.default import DefaultBotProperties  # noqa: E402
from aiogram.client.session.base import BaseSession  # noqa: E402
from aiogram.enums import ParseMode  # noqa: E402
from aiogram.fsm.storage.memory import MemoryStorage  # noqa: E402
from aiogram.types import (  # noqa: E402
    CallbackQuery,
    Chat,
    Message,
    Update,
    User,
)

from bot.config import Config  # noqa: E402
from bot.handlers import build_router  # noqa: E402
from bot.storage import Storage  # noqa: E402

USER = User(id=5, is_bot=False, first_name="Тест", username="tester")
CHAT = Chat(id=5, type="private")


class FakeSession(BaseSession):
    """Ничего никуда не шлёт — только записывает, что бот собирался отправить."""

    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        name = type(method).__name__
        self.calls.append((name, method))
        if name in ("SendMessage", "SendPhoto", "SendDocument"):
            return Message(
                message_id=1000 + len(self.calls),
                date=datetime.now(timezone.utc),
                chat=CHAT,
                from_user=User(id=7, is_bot=True, first_name="bot", username="b"),
                text=getattr(method, "text", None),
                caption=getattr(method, "caption", None),
            )
        if name == "GetMe":
            return User(id=7, is_bot=True, first_name="bot", username="testbot")
        return True

    async def stream_content(self, *args, **kwargs):  # pragma: no cover
        yield b""

    def last(self, kind=None):
        for name, method in reversed(self.calls):
            if kind is None or name == kind:
                return name, method
        return None, None


_uid = [0]


def _next_id():
    _uid[0] += 1
    return _uid[0]


def msg(text):
    return Update(
        update_id=_next_id(),
        message=Message(
            message_id=_next_id(),
            date=datetime.now(timezone.utc),
            chat=CHAT,
            from_user=USER,
            text=text,
        ),
    )


def cb(data, message_id=1001):
    return Update(
        update_id=_next_id(),
        callback_query=CallbackQuery(
            id=str(_next_id()),
            from_user=USER,
            chat_instance="x",
            data=data,
            message=Message(
                message_id=message_id,
                date=datetime.now(timezone.utc),
                chat=CHAT,
                from_user=User(id=7, is_bot=True, first_name="bot", username="b"),
                text="—",
            ),
        ),
    )


def check(condition, label):
    mark = "OK " if condition else "FAIL"
    print(f"  [{mark}] {label}")
    if not condition:
        check.failed += 1


check.failed = 0


async def main():
    tmp = Path(tempfile.mkdtemp(prefix="todorxoi-smoke-"))
    config = Config.from_env()
    config.db_path = tmp / "db.sqlite3"
    config.jsonl_path = tmp / "events.jsonl"
    config.validate()

    storage = Storage(config.db_path, config.jsonl_path)
    storage.connect()

    session = FakeSession()
    bot = Bot(
        token="123456:TEST",
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp["storage"] = storage
    dp["config"] = config
    dp.include_router(build_router())

    print("\n0. Окружение")
    from core.todo_image import SHAPING_ENGINE, SHAPING_OK, shaping_status

    check(SHAPING_OK, f"буквы тодо бичиг соединяются — {shaping_status()}")
    check(
        SHAPING_ENGINE == "harfbuzz",
        "используется HarfBuzz напрямую (не зависит от сборки Pillow)",
    )

    print("\n1. Команды и меню")
    await dp.feed_update(bot, msg("/start"))
    _, m = session.last("SendMessage")
    check(m and "Тодо Бичиг бот" in m.text, "/start отвечает приветствием")

    await dp.feed_update(bot, msg("/help"))
    _, m = session.last("SendMessage")
    check("Как пользоваться" in m.text, "/help описывает логику")

    await dp.feed_update(bot, msg("/mode"))
    _, m = session.last("SendMessage")
    check("Транслитерация" in m.text, "/mode показывает режим по умолчанию")

    print("\n2. Транслитерация (п.1)")
    await dp.feed_update(bot, msg("Хальмг улс"))
    _, m = session.last("SendMessage")
    check("Транслитерация" in m.text, "кириллица обработана в режиме translit")
    check("⏱" in m.text, "в ответе есть время работы")
    check(m.reply_markup is not None, "под ответом есть кнопки")

    print("\n3. Тодо бичиг (п.2)")
    await dp.feed_update(bot, msg("/todo хальмг улс"))
    _, m = session.last("SendMessage")
    check("Тодо бичиг" in m.text, "команда с текстом сразу обработана")
    check(any("᠀" <= ch <= "᢯" for ch in m.text), "в ответе есть тодо бичиг")
    buttons = [b.text for row in m.reply_markup.inline_keyboard for b in row]
    check("👍" in buttons and "👎" in buttons, "есть кнопки 👍/👎")

    print("\n4. Картинка (п.3)")
    await dp.feed_update(bot, msg("/image хальмг улс"))
    name, m = session.last()
    check(name in ("SendPhoto", "SendDocument"), f"пришла картинка ({name})")
    check("⏱" in (m.caption or ""), "в подписи к картинке есть время работы")

    print("\n4a. Дефис: составное слово vs суффикс")
    from core.transliterate import _translit_token

    for word in ("келн-мелн", "көвүн-күүкн", "эк-эцк"):
        got, src = _translit_token(word)
        check(src == "split" and " " in got,
              f"составное слово разобрано по частям: {word} -> {got}")
    # «мелн» — эхо-слово, в словаре его нет: часть должна уйти в модель,
    # а не тянуть за собой весь токен
    check(_translit_token("келн-мелн")[0].split()[1] not in ("", "мелн"),
          "незнакомая часть составного слова переведена моделью")
    suffix, src_s = _translit_token("һазр-ән")
    check(src_s == "model",
          f"короткий суффикс через дефис НЕ разобран: һазр-ән -> {suffix}")
    plain, src_p = _translit_token("хальмг")
    check(src_p == "dict", f"обычное слово берётся из словаря: хальмг -> {plain}")

    from core.translit_todo import todo_to_translit, translit_to_todo

    check("\u202f" not in translit_to_todo(_translit_token("келн-мелн")[0]),
          "в составном слове широкий пробел, а не узкий неразрывный")
    check("\u202f" in translit_to_todo("γazar-yēn"),
          "у суффикса, наоборот, узкий неразрывный пробел")

    print("\n4a1. x/k не склеиваются с соседней буквой в диграф")
    check(translit_to_todo("bolxu") == "\u184b\u1846\u182f\u184d\u1847",
          f"bolxu -> {translit_to_todo('bolxu')} (ждём ᡋᡆᠯᡍᡇ, а не ᡋᡆᡀᡇ)")
    check("\u1840" not in translit_to_todo("ādoulxu"),
          "ᡀ (lh) больше не возникает там, где её нет")
    check("\u1857" not in translit_to_todo("angxāraq"),
          "ᡗ (gh) больше не возникает там, где её нет")

    print("\n4a2. Правило či -> ᡔᡅ")
    check(translit_to_todo("či") == "\u1854\u1845",
          "č перед краткой i пишется через ᡔ (U+1854)")
    check("\u1854" in translit_to_todo("arčīxu"),
          "č перед долгой ī — тоже через ᡔ")
    check("\u1852" in translit_to_todo("čōno") and "\u1854" not in translit_to_todo("čōno"),
          "перед другими гласными č остаётся ᡒ (U+1852)")
    check(todo_to_translit(translit_to_todo("abči")) == "abči",
          "обратно ᡔᡅ разворачивается в či")

    print("\n4a3. Буква k перед задней гласной, тире, знаки препинания")
    check(translit_to_todo("karou") == "\u1857\u1820\u1837\u1846\u1847",
          f"karou -> {translit_to_todo('karou')} (ждём ᡗᠠᠷᡆᡇ с TODO KA)")
    check("\u1857" not in translit_to_todo("kelen"),
          "перед передней гласной k остаётся ᡍ")
    check("\ufe31" in translit_to_todo("xalimaq — ulus"),
          "тире заменяется на вертикальное U+FE31")
    check("\u202f" in translit_to_todo("γazar-yēn"),
          "дефис внутри слова по-прежнему узкий неразрывный пробел")

    from core.punctuation import add_punctuation

    from core.punctuation import FRAME_MIN_CHARS, needs_frame

    check(not needs_frame("хальмг улс"), "короткий текст не обрамляется")
    check(needs_frame("х" * (FRAME_MIN_CHARS + 1)), "длинный — обрамляется")
    plain_short = add_punctuation(translit_to_todo("eke. ecege."), frame=False)
    check("\u1800" not in plain_short and "\u1805" not in plain_short,
          "без обрамления нет ни бирги, ни четырёх точек")
    check("\u1803" in plain_short, "а обычная точка на месте")

    marked = add_punctuation(translit_to_todo("xalimaq ulus, eke. ecege."))
    check(marked.startswith("\u1800"), "в начале текста стоит бирга")
    check("\u1802" in marked, "запятая заменена на ᠂")
    check("\u1803" in marked, "точка внутри текста — ᠃")
    check(marked.rstrip().endswith("\u1805"), "в конце текста — четыре точки ᠅")
    q = add_punctuation(translit_to_todo("yayu?"))
    check(q.rstrip().endswith("\ufe16\u1805"),
          "после вопросительного знака четыре точки добавляются, знак остаётся")

    print("\n4a4. Семейство шрифтов с кириллической записью")
    import core
    from core import ImageOptions
    from core.clear_script import FONT_FILES, available, font_path, translit_to_font
    from core.pipeline import FONTS

    check(available(), "файлы всех шрифтов семейства на месте")
    check(len(FONTS) == len(FONT_FILES) + 1,
          f"в настройках доступны все шрифты: {FONTS}")
    check(translit_to_font("xalimaq") == "ХалимаЩ",
          f"правила дают запись для шрифта: {translit_to_font('xalimaq')}")
    # запись у семейства общая, поэтому каждый шрифт должен нарисовать её
    for key in FONTS:
        r = core.process("хальмг", "image", ImageOptions(font=key))
        check(bool(r.image_size) and r.image_size[0] > 0,
              f"картинка шрифтом {key}: {r.image_size}")

    # Подгонка записи под начертание: отрезок стержня там, где буквы
    # наезжают друг на друга, и двойной пробел между словами.
    from core.clear_script import SPINE_MARK, WORD_SPACE, fit_to_font, gaps_for
    rec = translit_to_font("teyimü")
    check(rec == "тэйимю", f"тиим -> {rec}")
    for key in FONT_FILES:
        fitted = fit_to_font(rec, key)
        check(SPINE_MARK in fitted, f"{key}: стержень удлинён — {fitted}")
        check(gaps_for(key), f"{key}: таблица пар загрузилась")
    spaced = fit_to_font(translit_to_font("eke ecege"), "clear")
    check(WORD_SPACE in spaced and "  " in spaced,
          f"слова разведены двойным пробелом: {spaced}")
    check(fit_to_font(rec, "universal") == rec.replace(" ", WORD_SPACE),
          "для неизвестного шрифта запись не ломается")

    # знаки препинания должны доезжать и до кириллической записи
    marked = add_punctuation(translit_to_font("eke, ecege. \u0101bu?"))
    check(marked.startswith("\u1800"), "бирга ставится и в этой записи")
    check("\u1802" in marked and "\u1803" in marked
          and marked.rstrip().endswith("\u1805"),
          "запятая, точка и четыре точки — тоже")
    # fontTools нужен только инструментам из tools/, боту он не нужен —
    # поэтому проверка мягкая
    try:
        from fontTools.ttLib import TTFont as _TTFont

        from tools.add_marks_to_fonts import MARKS as _MARKS

        _drawn = set(_MARKS.values())
        for key in FONT_FILES:
            ft = _TTFont(str(font_path(key)))
            cmap, glyf, hm = ft.getBestCmap(), ft["glyf"], ft["hmtx"]
            check(all(cp in cmap for cp in _MARKS),
                  f"{key}: все дорисованные глифы на месте")
            # Знак не должен налезать на хвост предыдущей буквы. Пробела в
            # тексте нет: отступ заложен слева от чернил самого глифа, и он
            # обязан перекрывать самый большой вынос чернил букв за шаг.
            overhang = max(glyf[g].xMax - hm[g][0]
                           for g in ft.getGlyphOrder()
                           if g not in _drawn and glyf[g].numberOfContours > 0)
            for cp, title in ((0x1802, "запятой"), (0x1803, "точки"),
                              (0xFE16, "вопроса")):
                lsb = glyf[cmap[cp]].xMin
                check(lsb > overhang,
                      f"{key}: отступ у {title} {lsb} больше выноса {overhang}")
    except ImportError:
        print("  [--] fontTools не установлен, проверку глифов пропускаю")

    print("\n4a5. Регистр кириллицы не меняет результат")
    for pair in (("Җирһл", "җирһл"), ("Хальмг улс", "хальмг улс")):
        big, small = (core.process(w, "todo").todo for w in pair)
        check(big == small, f"{pair[0]} и {pair[1]} -> {big}")

    print("\n4a6. Длинный текст — несколько картинок по порядку")
    from core import MAX_INPUT_CHARS

    check(MAX_INPUT_CHARS == 10000, f"лимит входа: {MAX_INPUT_CHARS}")
    long_text = ("Хальмг улс мана һазр деер бәәнә. Эн бичг ут болх, учрн "
                 "биргиг болн дөрвн цегиг зөвәр ут бичгт тәвнәвидн. " * 40)
    long_text = long_text[:MAX_INPUT_CHARS]
    res = core.process(long_text, "image")
    check(len(res.pages) > 1, f"текст разложен на листы: {len(res.pages)}")
    check(not res.pages_dropped, "все листы поместились")
    check(res.image is res.pages[0][0], "res.image — первый лист")
    # ширина каждого листа в пределах допустимого, высота одинаковая
    from core.todo_image import MAX_SIDE

    widths = [size[0] for _buf, size in res.pages]
    heights = {size[1] for _buf, size in res.pages}
    check(max(widths) <= MAX_SIDE, f"ширина листа не больше {MAX_SIDE}: {max(widths)}")
    check(len(heights) == 1, f"высота у всех листов одна: {heights}")

    before = len(session.calls)
    await dp.feed_update(bot, msg(long_text))
    sent = session.calls[before:]
    photos = [m for name, m in sent if name in ("SendPhoto", "SendDocument")]
    check(len(photos) == len(res.pages),
          f"отправлено столько же сообщений, сколько листов: {len(photos)}")
    check(f"картинка 1 из {len(photos)}" in (photos[0].caption or ""),
          f"первый лист подписан: {(photos[0].caption or '')[:60]}")
    check(f"картинка {len(photos)} из {len(photos)}" in (photos[-1].caption or ""),
          "последний лист подписан")
    check(photos[-1].reply_markup and not photos[0].reply_markup,
          "кнопки 👍/👎 — только под последним листом")

    # короткому тексту номера не нужны
    await dp.feed_update(bot, msg("хальмг"))
    _name, m = session.last()
    check("из" not in (m.caption or "").split("\n")[0],
          f"одна картинка — без нумерации: {(m.caption or '').split(chr(10))[0]}")

    print("\n4b. Настройки картинки")
    await dp.feed_update(bot, msg("/settings"))
    _, m = session.last("SendMessage")
    check("Настройки картинки" in m.text, "/settings показывает текущие настройки")

    await dp.feed_update(bot, cb("set:set:bg:transparent"))
    saved = await storage.get_settings(USER.id)
    check(saved.get("bg") == "transparent", "прозрачный фон сохранён в БД")

    await dp.feed_update(bot, cb("set:set:fg:red"))
    await dp.feed_update(bot, msg("/image хальмг"))
    name, m = session.last()
    check(name == "SendDocument", f"с прозрачным фоном картинка ушла файлом ({name})")

    await dp.feed_update(bot, cb("set:set:bg:red"))  # фон = цвет текста
    await dp.feed_update(bot, msg("/image хальмг"))
    name, m = session.last()
    check("совпал с фоном" in (m.caption or ""), "про откат по цветам сказано в подписи")
    check(name == "SendPhoto", "после отката фон непрозрачный — снова фото")

    await dp.feed_update(bot, cb("set:reset:-"))
    saved = await storage.get_settings(USER.id)
    check(not saved, "сброс настроек очищает запись")

    print("\n5. Фидбэк: 👍")
    row = await storage.get_request(1)
    await dp.feed_update(bot, cb(f"fb:up:{row['id']}"))
    stats = await storage.stats()
    check(stats["up"] == 1, "палец вверх записан в базу")

    print("\n6. Фидбэк: 👎 и исправление")
    await dp.feed_update(bot, cb("fb:down:2"))
    _, m = session.last("SendMessage")
    check("правильный ответ" in m.text, "после 👎 бот просит правильный вариант")

    await dp.feed_update(bot, msg("xalimaq ulus"))
    _, m = session.last("SendMessage")
    check("исправление сохранил" in m.text, "исправление принято")
    stats = await storage.stats()
    check(stats["corrections"] == 1, "исправление лежит в базе")
    check(stats["down"] == 1, "минус записан")

    print("\n7. Исправление не перехватывает обычный текст дальше")
    await dp.feed_update(bot, msg("хальмг"))
    # режим на этот момент — «картинка» (его включили в п.4), поэтому
    # смотрим на последний вызов любого типа, а не только SendMessage
    name, m = session.last()
    body = getattr(m, "text", None) or getattr(m, "caption", None) or ""
    check("⏱" in body, f"следующее сообщение снова идёт в перевод ({name})")

    print("\n8. /cancel")
    await dp.feed_update(bot, cb("fb:down:1"))
    await dp.feed_update(bot, msg("/cancel"))
    _, m = session.last("SendMessage")
    check("Минус записан" in m.text, "/cancel выходит из ожидания исправления")

    print("\n9. Кнопка «→ Картинка» под готовым ответом")
    before = len(session.calls)
    await dp.feed_update(bot, cb("conv:image:1"))
    name, m = session.last()
    check(name in ("SendPhoto", "SendDocument"), "кнопка конвертации отдала картинку")
    check(len(session.calls) > before, "запрос действительно обработан")

    print("\n10. Админские команды")
    await dp.feed_update(bot, msg("/stats"))
    _, m = session.last("SendMessage")
    check("Статистика" in m.text, "/stats доступен админу")
    name, _ = None, None
    await dp.feed_update(bot, msg("/export corrections"))
    name, m = session.last()
    check(name == "SendDocument", "/export отдаёт CSV файлом")

    print("\n11. Мусор на входе")
    await dp.feed_update(bot, msg("12345"))
    _, m = session.last("SendMessage")
    check("Не удалось распознать" in m.text, "непонятный текст — понятная ошибка")

    await dp.feed_update(bot, msg("/nosuchcommand"))
    _, m = session.last("SendMessage")
    check("Не знаю такой команды" in m.text, "неизвестная команда не уходит в перевод")

    storage.close()
    await bot.session.close()

    print()
    if check.failed:
        print(f"❌ провалено проверок: {check.failed}")
        sys.exit(1)
    print("✅ все проверки прошли")


if __name__ == "__main__":
    asyncio.run(main())
