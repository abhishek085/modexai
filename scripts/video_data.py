"""Pull real ticket-triage measurements from the recipe index into the launch video's props."""

from __future__ import annotations

import json
from pathlib import Path

from modexai import recipes

ROOT = Path(__file__).resolve().parents[1]
BAR = 0.8


def main() -> None:
    rows = {}
    for r in recipes.load_index(str(ROOT / "src/modexai/data/recipes.json")):
        if r.get("eval", {}).get("suite") == "ticket-triage@1" and r["kind"] == "benchmark":
            rows[r["model"]["tag"]] = {"tag": r["model"]["tag"], "score": r["eval"]["score"], "p50": r["eval"]["p50_ms"],
                                       "size": r["model"].get("size_gb") or 0, "hw": r["hardware"]["label"]}
    ranked = sorted(rows.values(), key=lambda x: x["size"])
    passing = [x for x in ranked if x["score"] >= BAR]
    hire = passing[0]
    # A readable race: the smallest models that miss, the hire, and a bigger model that also passes.
    smaller = [x for x in ranked if x["size"] < hire["size"] and x["score"] < BAR][-2:]
    bigger = [x for x in passing if x["size"] > hire["size"]][-1:]
    race = smaller + [hire] + bigger
    data = {"race": [{k: v for k, v in x.items() if k != "hw"} for x in race],
            "hire": {k: v for k, v in hire.items() if k != "hw"},
            "suite": "ticket-triage", "hardware": hire["hw"], "bar": BAR}
    out = ROOT / "video/src/data.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
