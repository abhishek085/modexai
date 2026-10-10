"""Assemble the static exchange site for GitHub Pages into ./_site."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from modexai import config
from modexai.suites import list_suites

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(config.WEB_DIR, SITE)
    shutil.copy(config.DATA_DIR / "registry.json", SITE / "registry.json")
    shutil.copy(config.DATA_DIR / "recipes.json", SITE / "recipes.json")
    shutil.copy(config.DATA_DIR / "notes.json", SITE / "notes.json")
    (SITE / "suites.json").write_text(json.dumps({"suites": [s.summary() for s in list_suites()]}, indent=2), encoding="utf-8")
    for name in ("llms.txt",):
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, SITE / name)
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site built in {SITE}")


if __name__ == "__main__":
    main()
