"""Print the README's "what we measured" table from the recipe index (so numbers are never hand-typed)."""

from __future__ import annotations

from collections import defaultdict

from modexai import recipes
from modexai.suites import list_suites

BAR = 0.8


def main() -> None:
    by_suite: dict[str, list[dict]] = defaultdict(list)
    hw = set()
    for r in recipes.load_index():
        if r.get("eval") and r["kind"] == "benchmark":
            by_suite[r["eval"]["suite"].split("@")[0]].append(r)
            hw.add(r["hardware"]["label"])
    names = {s.id: s.name for s in list_suites()}
    print(f"| Job | Hire (smallest ≥ {BAR:.0%}) | Score | p50 | Size | Best overall | Models tested |")
    print("|---|---|---|---|---|---|---|")
    for sid in sorted(by_suite, key=lambda s: names.get(s, s)):
        rs = by_suite[sid]
        best = max(rs, key=lambda r: (r["eval"]["score"], -(r["model"].get("size_gb") or 99)))
        passing = [r for r in rs if r["eval"]["score"] >= BAR]
        hire = min(passing, key=lambda r: (r["model"].get("size_gb") or 99, -r["eval"]["score"])) if passing else None
        h = (f"`{hire['model']['tag']}` | {hire['eval']['score']:.0%} | {hire['eval']['p50_ms']:.0f} ms | "
             f"{hire['model'].get('size_gb')} GB") if hire else "— none yet: cloud earns its place | | |"
        print(f"| {names.get(sid, sid)} | {h} | `{best['model']['tag']}` {best['eval']['score']:.0%} | {len(rs)} |")
    print(f"\n_Measured on {', '.join(sorted(hw))}._")


if __name__ == "__main__":
    main()
