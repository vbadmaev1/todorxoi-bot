"""Build a private Kaggle notebook with the prepared spelling pairs embedded."""

from base64 import b64encode
from pathlib import Path
import gzip
import json

root = Path(__file__).parent
out = root / "kaggle"
out.mkdir(exist_ok=True)

payload = b64encode(gzip.compress((root / "pairs.tsv").read_bytes(), mtime=0)).decode()
source = (
    "from base64 import b64decode\n"
    "from pathlib import Path\n"
    "import gzip\n"
    "import sys\n\n"
    "sys.path.insert(0, '/kaggle/working')\n"
    f"Path('/kaggle/working/spelling.py').write_text({(root / 'spelling.py').read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
    "data = Path('/kaggle/working/pairs.tsv')\n"
    f"data.write_bytes(gzip.decompress(b64decode({payload!r})))\n"
    "sys.argv = ['train.py', str(data), '/kaggle/working/model']\n\n"
    + (root / "train.py").read_text(encoding="utf-8")
)
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
    "cells": [{"id": "train", "cell_type": "code", "execution_count": None,
               "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
}
metadata = {
    "id": "vbadmaev1/todorxoi-todo-hudum", "title": "Todorxoi Todo Hudum",
    "code_file": "todo_hudum.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
}
(out / "todo_hudum.ipynb").write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
(out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
