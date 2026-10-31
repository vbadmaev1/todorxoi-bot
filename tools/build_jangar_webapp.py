# -*- coding: utf-8 -*-
"""Собрать мини-приложение «Джангар» в webapp/jangar/ из читалки research/jangar/web.

  python tools/build_jangar_webapp.py [путь к research/jangar/web]

research/jangar (OCR скана, сопоставление, словарь Тодаевой) в репозиторий не входит — оттуда берётся только
готовая читалка: index.html и data/*.json как есть, сканы pages/*.jpg пережимаются в WebP (качество 60:
на глаз то же, вдвое легче). Рядом с каждым JSON кладётся .gz — aiohttp отдаёт его браузеру, который умеет
gzip (главы весят 0,1–1 МБ, сжатые — впятеро меньше). Раздаёт папку бот: bot/webapp.py.
"""
import gzip
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "webapp" / "jangar"
DEFAULT_SRC = ROOT / "research" / "jangar" / "web"


def main():
    src = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_SRC
    if not (src / "index.html").exists():
        sys.exit(f"нет читалки: {src}/index.html")
    (DST / "data").mkdir(parents=True, exist_ok=True)
    (DST / "pages").mkdir(parents=True, exist_ok=True)

    html = (src / "index.html").read_text()
    html = html.replace("`pages/p${pad(no, 3)}.jpg`", "`pages/p${pad(no, 3)}.webp`")
    (DST / "index.html").write_text(html)

    for f in sorted((src / "data").glob("*.json")):
        shutil.copy2(f, DST / "data" / f.name)
        with gzip.open(DST / "data" / (f.name + ".gz"), "wb", compresslevel=9) as out:
            out.write(f.read_bytes())

    n = 0
    for f in sorted((src / "pages").glob("p*.jpg")):
        out = DST / "pages" / (f.stem + ".webp")
        if not out.exists() or out.stat().st_mtime < f.stat().st_mtime:
            Image.open(f).convert("RGB").save(out, "WEBP", quality=60, method=6)
            n += 1
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    print(f"{DST}: {n} страниц пережато, всего {size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
