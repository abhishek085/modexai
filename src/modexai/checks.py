"""Deterministic graders. No LLM-as-judge: every check here is reproducible and explainable.

A check is a dict like ``{"type": "label", "expected": "billing", "labels": [...]}``. ``grade`` returns
``(passed, detail)`` where detail says why, so a failing case can be read by a human or an agent.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.S)


def clean(text: str) -> str:
    """Strip reasoning blocks some small models emit before answering."""
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text.lower():  # unterminated think block: keep what follows the last tag
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Any | None:
    """Return the first JSON object/array found in text (fenced or bare)."""
    text = clean(text)
    candidates = [m.strip() for m in _FENCE_RE.findall(text)] + [text]
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def extract_number(text: str) -> float | None:
    text = clean(text)
    m = re.search(r"(?:answer|final answer|result)\s*[:=]?\s*\$?(-?[\d,]*\.?\d+)", text, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?\$?[\d,]*\.?\d+", text)
    for raw in reversed(nums):
        try:
            return float(raw.replace(",", "").replace("$", ""))
        except ValueError:
            continue
    return None


def extract_sql(text: str) -> str:
    text = clean(text)
    fenced = _FENCE_RE.findall(text)
    body = fenced[0] if fenced else text
    m = re.search(r"\b(with|select)\b.*", body, re.I | re.S)
    sql = m.group(0) if m else body
    return sql.split(";")[0].strip()


def find_label(text: str, labels: list[str]) -> str | None:
    """Find which allowed label the model chose.

    Prefers an explicit ``Label: x`` / ``Answer: x`` line, then the *last* label mentioned (models often
    reason first, answer last). Longer labels win ties so "malicious-instruction" beats "malicious".
    """
    text = clean(text)
    by_len = sorted(labels, key=len, reverse=True)
    m = re.findall(r"(?:label|answer|category|classification|verdict|intent)\s*[:=]\s*\**\s*([^\n*]+)", text, re.I)
    if m:
        tail = _norm(m[-1])
        for lab in by_len:
            if tail.startswith(_norm(lab)):
                return lab
    low = _norm(text)
    best: tuple[int, str] | None = None
    for lab in by_len:
        for hit in re.finditer(rf"(?<![a-z0-9]){re.escape(_norm(lab))}(?![a-z0-9])", low):
            if best is None or hit.start() > best[0]:
                best = (hit.start(), lab)
    return best[1] if best else None


def _values_match(got: Any, want: Any, tol: float = 0.01) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return str(got).lower() == str(want).lower()
    if isinstance(want, (int, float)):
        try:
            g = float(str(got).replace(",", "").replace("$", ""))
        except ValueError:
            return False
        return abs(g - want) <= max(tol, abs(want) * tol)
    if isinstance(want, list):
        if not isinstance(got, list):
            return False
        return sorted(_norm(x) for x in got) == sorted(_norm(x) for x in want)
    if want is None:
        return got in (None, "", "null", "none", "n/a")
    return _norm(got) == _norm(want)


def token_f1(a: str, b: str) -> float:
    ta, tb = _norm(a).split(), _norm(b).split()
    if not ta or not tb:
        return float(ta == tb)
    common: dict[str, int] = {}
    for t in ta:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in tb:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


def _grade_label(out: str, c: dict) -> tuple[bool, str]:
    labels = c.get("labels") or [c["expected"]]
    got = find_label(out, labels)
    accept = c.get("accept") or [c["expected"]]
    ok = got is not None and any(_norm(got) == _norm(a) for a in accept)
    return ok, f"label={got!r} expected={'|'.join(accept)}"


def _grade_json_fields(out: str, c: dict) -> tuple[bool, str]:
    obj = extract_json(out)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        return False, "no JSON object in output"
    lowered = {_norm(k).replace(" ", "_"): v for k, v in obj.items()}
    bad = []
    for k, want in c["expected"].items():
        got = obj.get(k, lowered.get(_norm(k).replace(" ", "_"), "<missing>"))
        if not _values_match(got, want, c.get("tolerance", 0.01)):
            bad.append(f"{k}: got {got!r} want {want!r}")
    return not bad, "; ".join(bad) or "all fields match"


def _grade_number(out: str, c: dict) -> tuple[bool, str]:
    got = extract_number(out)
    want = float(c["expected"])
    ok = got is not None and abs(got - want) <= max(c.get("tolerance", 0.01), abs(want) * 1e-6)
    return ok, f"number={got} expected={want}"


def _grade_contains(out: str, c: dict) -> tuple[bool, str]:
    low = clean(out).lower()
    missing = [s for s in c.get("all", []) if s.lower() not in low]
    present = [s for s in c.get("none", []) if s.lower() in low]
    any_of = c.get("any", [])
    any_ok = not any_of or any(s.lower() in low for s in any_of)
    ok = not missing and not present and any_ok
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if present:
        parts.append(f"leaked {present}")
    if not any_ok:
        parts.append(f"none of {any_of}")
    return ok, "; ".join(parts) or "ok"


def _grade_regex(out: str, c: dict) -> tuple[bool, str]:
    ok = re.search(c["pattern"], clean(out), re.I | re.S) is not None
    return ok, f"/{c['pattern']}/ {'matched' if ok else 'no match'}"


def _grade_sql(out: str, c: dict) -> tuple[bool, str]:
    """Execution accuracy: run the model's query and the gold query on an in-memory SQLite DB."""
    sql = extract_sql(out)
    if not re.match(r"^\s*(with|select)\b", sql, re.I):
        return False, "no SELECT statement found"
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(c["setup"])
        con.execute("PRAGMA query_only = ON")
        want = con.execute(c["gold"]).fetchall()
        try:
            got = con.execute(sql).fetchall()
        except sqlite3.Error as e:
            return False, f"SQL error: {e}"
    finally:
        con.close()

    def canon(rows: list[tuple]) -> list[tuple]:
        return sorted(tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in rows)

    ordered = c.get("ordered", False)
    ok = (got == want) if ordered else (canon(got) == canon(want))
    return ok, f"{len(got)} rows {'match' if ok else 'differ from gold'}"


def _grade_tool_call(out: str, c: dict) -> tuple[bool, str]:
    obj = extract_json(out)
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return False, "no JSON tool call in output"
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    want = c["expected"]
    if name != want["name"]:
        return False, f"called {name!r}, expected {want['name']!r}"
    bad = [
        f"{k}: got {args.get(k)!r} want {v!r}"
        for k, v in want.get("arguments", {}).items()
        if not _values_match(args.get(k), v)
    ]
    return not bad, "; ".join(bad) or "correct tool and arguments"


def _grade_similar(out: str, c: dict) -> tuple[bool, str]:
    f1 = token_f1(clean(out), c["expected"])
    thr = c.get("threshold", 0.6)
    return f1 >= thr, f"token-F1={f1:.2f} (threshold {thr})"


def _grade_exact(out: str, c: dict) -> tuple[bool, str]:
    ok = _norm(clean(out)) == _norm(c["expected"])
    return ok, "exact match" if ok else "differs"


GRADERS = {
    "label": _grade_label,
    "json_fields": _grade_json_fields,
    "number": _grade_number,
    "contains": _grade_contains,
    "regex": _grade_regex,
    "sql_exec": _grade_sql,
    "tool_call": _grade_tool_call,
    "similar": _grade_similar,
    "exact": _grade_exact,
}


def grade(output: str, check: dict) -> tuple[bool, str]:
    fn = GRADERS.get(check.get("type", ""))
    if fn is None:
        raise ValueError(f"unknown check type {check.get('type')!r}; known: {sorted(GRADERS)}")
    try:
        return fn(output, check)
    except Exception as e:  # a grader must never crash a benchmark run
        return False, f"grader error: {e}"


def infer_check(expected: Any, labels: list[str] | None = None) -> dict:
    """Pick a sensible check for a user-supplied (or shadow-recorded) expected answer."""
    if isinstance(expected, dict):
        return {"type": "json_fields", "expected": expected}
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return {"type": "number", "expected": expected}
    s = str(expected).strip()
    parsed = extract_json(s) if s[:1] in "{[" else None
    if isinstance(parsed, dict):
        return {"type": "json_fields", "expected": parsed}
    if labels:
        return {"type": "label", "expected": s, "labels": labels}
    if re.fullmatch(r"-?\$?[\d,]*\.?\d+", s):
        return {"type": "number", "expected": float(s.replace(",", "").replace("$", ""))}
    if len(s.split()) <= 4:
        return {"type": "label", "expected": s, "labels": [s]}
    return {"type": "similar", "expected": s, "threshold": 0.6}
