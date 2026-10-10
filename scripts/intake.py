"""GitHub Action helper: turn a recipe issue into a file under registry/recipes/.

Reads the issue from $GITHUB_EVENT_PATH (never from interpolated shell strings), validates the recipe,
and writes it. The issue text is treated purely as data: nothing in it is executed.
Writes a markdown summary to $INTAKE_SUMMARY and sets `recipe_id` / `ok` step outputs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from modexai import recipes

ROOT = Path(__file__).resolve().parents[1]


def out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            fh.write(f"{key}={value}\n")


def main() -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    issue = event.get("issue") or {}
    body = issue.get("body") or ""
    summary_path = Path(os.environ.get("INTAKE_SUMMARY", "intake-summary.md"))
    d = recipes.extract_from_issue(body)
    if d is None:
        summary_path.write_text("I couldn't find a ```json block with a recipe in this issue. "
                                "Paste the output of `modexai recipes show <id>` or use `modexai recipes share`.")
        out("ok", "false")
        return 0
    # The id is content-derived; recompute so a hand-edited recipe still gets a correct id.
    if "id" not in d:
        d["id"] = recipes.recipe_id(d)
    problems = recipes.validate(d)
    if problems:
        summary_path.write_text("This recipe didn't pass validation:\n\n" + "\n".join(f"- {p}" for p in problems))
        out("ok", "false")
        return 0
    dest = ROOT / "registry" / "recipes" / f"{d['id']}.json"
    if dest.exists():
        summary_path.write_text(f"Recipe `{d['id']}` is already in the index — thanks for confirming it!")
        out("ok", "false")
        return 0
    author = (issue.get("user") or {}).get("login")
    if author and not d.get("author"):
        d["author"] = author  # author is excluded from the content id
    dest.write_text(json.dumps(d, indent=2) + "\n")
    ev = d.get("eval") or d.get("private_eval")
    summary_path.write_text(
        f"Valid recipe `{d['id']}`: **{d['model']['tag']}** scored **{ev['score']:.0%}** on "
        f"`{d.get('eval', {}).get('suite', d['task'])}` · {d['hardware'].get('label', '?')}. "
        "A maintainer will review the pull request."
    )
    out("ok", "true")
    out("recipe_id", d["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
