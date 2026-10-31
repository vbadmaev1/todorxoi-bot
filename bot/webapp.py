# -*- coding: utf-8 -*-
"""Веб-сервер мини-приложений в том же процессе, что и бот (aiohttp приходит вместе с aiogram).

Бот работает long polling'ом и сам по себе порт не слушает; мини-приложение «Джангар» (/jangar) — статическая
страница webapp/jangar/ (собирается tools/build_jangar_webapp.py), и Telegram открывает её по https-адресу.
Хостинг собирает образ из репозитория и выставляет порт контейнера наружу по https — этот адрес плюс «/jangar/»
и есть JANGAR_URL.

  /jangar/   — мини-приложение (index.html, data/*.json с готовыми .gz, pages/*.webp)
  /healthz   — «ok», для проверки хостинга
"""

import logging
from pathlib import Path

from aiohttp import web

log = logging.getLogger(__name__)

WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"


@web.middleware
async def cache_headers(request: web.Request, handler):
    resp = await handler(request)
    # сканы не меняются — пусть лежат в кэше; страница и данные пересобираются — проверять каждый раз
    if request.path.startswith("/jangar/pages/"):
        resp.headers["Cache-Control"] = "public, max-age=604800"
        # у aiohttp своя таблица типов, .webp в ней нет: без этого скан уходит как application/octet-stream
        if request.path.endswith(".webp"):
            resp.content_type = "image/webp"
    else:
        resp.headers["Cache-Control"] = "no-cache"
    return resp


async def health(_: web.Request) -> web.Response:
    return web.Response(text="ok")


async def jangar_index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(WEBAPP_DIR / "jangar" / "index.html")


def build_app() -> web.Application:
    app = web.Application(middlewares=[cache_headers])
    app.router.add_get("/healthz", health)
    app.router.add_get("/jangar", lambda _: web.HTTPFound("/jangar/"))
    app.router.add_get("/jangar/", jangar_index)
    # FileResponse сам отдаёт соседний .gz, если браузер принимает gzip
    app.router.add_static("/jangar/", WEBAPP_DIR / "jangar", show_index=False)
    return app


async def start(port: int) -> "web.AppRunner | None":
    """Запустить сервер на 0.0.0.0:port; без собранного webapp/jangar — не запускаем (бот работает и так)."""
    if not (WEBAPP_DIR / "jangar" / "index.html").exists():
        log.warning("мини-приложение не собрано (нет webapp/jangar/index.html) — веб-сервер не запущен")
        return None
    runner = web.AppRunner(build_app(), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    log.info("мини-приложения: http://0.0.0.0:%d/jangar/", port)
    return runner
