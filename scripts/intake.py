"""GitHub Action helper: turn a recipe or note issue into a file under registry/.

Reads the issue from $GITHUB_EVENT_PATH (never from interpolated shell strings), validates the recipe or
note, and writes it. The issue text is treated purely as data: nothing in it is executed.
Writes a markdown summary to $INTAKE_SUMMARY and sets `kind` / `item_id` / `ok` step outputs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from modexai import notes, recipes

ROOT = Path(__file__).resolve().parents[1]


def out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")


def intake_note(d: dict, author: str | None, summary_path: Path) -> int:
    if author and not d.get("author"):
        d["author"] = author  # the GitHub account that posted it; excluded from the note id
    problems = notes.validate(d)
    if problems:
        summary_path.write_text("This note didn't pass validation:\n\n" + "\n".join(f"- {p}" for p in problems),
                                encoding="utf-8")
        out("ok", "false")
        return 0
    dest = ROOT / "registry" / "notes" / f"{d['id']}.json"
    if dest.exists():
        summary_path.write_text(f"Note `{d['id']}` is already in the index.", encoding="utf-8")
        out("ok", "false")
        return 0
    dest.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(f"Valid note `{d['id']}` from **{d['agent']}** on `{d['on']}`. "
                            "A maintainer will review the pull request.", encoding="utf-8")
    out("ok", "true")
    out("kind", "note")
    out("item_id", d["id"])
    return 0


def main() -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    issue = event.get("issue") or {}
    body = issue.get("body") or ""
    author = (issue.get("user") or {}).get("login")
    summary_path = Path(os.environ.get("INTAKE_SUMMARY", "intake-summary.md"))
    d = recipes.extract_from_issue(body)
    if d is None:
        summary_path.write_text("I couldn't find a ```json block with a recipe or note in this issue. "
                                "Use `modexai recipes share` or `modexai notes post`.", encoding="utf-8")
        out("ok", "false")
        return 0
    if d.get("schema") == notes.SCHEMA:
        return intake_note(d, author, summary_path)
    # The id is content-derived; recompute so a hand-edited recipe still gets a correct id.
    if "id" not in d:
        d["id"] = recipes.recipe_id(d)
    problems = recipes.validate(d)
    if problems:
        summary_path.write_text("This recipe didn't pass validation:\n\n" + "\n".join(f"- {p}" for p in problems), encoding="utf-8")
        out("ok", "false")
        return 0
    dest = ROOT / "registry" / "recipes" / f"{d['id']}.json"
    if dest.exists():
        summary_path.write_text(f"Recipe `{d['id']}` is already in the index — thanks for confirming it!", encoding="utf-8")
        out("ok", "false")
        return 0
    if author and not d.get("author"):
        d["author"] = author  # author is excluded from the content id
    dest.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    ev = d.get("eval") or d.get("private_eval")
    summary_path.write_text(
        f"Valid recipe `{d['id']}`: **{d['model']['tag']}** scored **{ev['score']:.0%}** on "
        f"`{d.get('eval', {}).get('suite', d['task'])}` · {d['hardware'].get('label', '?')}. "
        "A maintainer will review the pull request.", encoding="utf-8"
    )
    out("ok", "true")
    out("kind", "recipe")
    out("item_id", d["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
