from modexai.checks import extract_json, extract_number, extract_sql, find_label, grade, infer_check, token_f1


def test_label_prefers_explicit_label_line():
    out = "Reasoning: looks like an attack.\nLabel: malicious-instruction"
    assert find_label(out, ["safe", "suspicious", "malicious-instruction"]) == "malicious-instruction"


def test_label_does_not_match_inside_other_words():
    assert find_label("unsafe", ["safe", "unsafe"]) == "unsafe"
    assert find_label("This is safe.", ["safe", "unsafe"]) == "safe"


def test_label_takes_last_mention_and_strips_think():
    out = "<think>maybe billing? no</think>It is not billing, it's technical."
    assert find_label(out, ["billing", "technical"]) == "technical"


def test_label_accepts_alternatives():
    c = {"type": "label", "expected": "malicious-instruction", "accept": ["malicious-instruction", "suspicious"],
         "labels": ["safe", "suspicious", "malicious-instruction"]}
    assert grade("Label: Suspicious", c)[0]
    assert not grade("Label: safe", c)[0]


def test_extract_json_fenced_and_embedded():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert extract_json("no json here") is None


def test_json_fields_numbers_and_case():
    c = {"type": "json_fields", "expected": {"total": 1250.0, "vendor": "Northwind Traders", "email": None}}
    assert grade('{"total": "1,250.00", "vendor": "northwind traders", "email": null}', c)[0]
    ok, why = grade('{"total": 125, "vendor": "Northwind Traders", "email": null}', c)
    assert not ok and "total" in why


def test_number_extraction():
    assert extract_number("work... Answer: 1,350") == 1350
    assert extract_number("The price is $30.") == 30
    assert grade("so 36", {"type": "number", "expected": 36})[0]


SETUP = "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1),(2),(3);"


def test_sql_exec_compares_results_not_text():
    c = {"type": "sql_exec", "setup": SETUP, "gold": "SELECT COUNT(*) FROM t"}
    assert grade("```sql\nSELECT count(x) FROM t;\n```", c)[0]
    assert not grade("SELECT 1", c)[0]
    assert not grade("DROP TABLE t", c)[0]


def test_sql_exec_is_read_only():
    c = {"type": "sql_exec", "setup": SETUP, "gold": "SELECT COUNT(*) FROM t"}
    ok, why = grade("WITH a AS (SELECT 1) SELECT * FROM a; DELETE FROM t", c)
    assert extract_sql("WITH a AS (SELECT 1) SELECT * FROM a; DELETE FROM t").endswith("FROM a")
    assert not ok


def test_tool_call_shapes():
    c = {"type": "tool_call", "expected": {"name": "set_timer", "arguments": {"minutes": 25}}}
    assert grade('{"name": "set_timer", "arguments": {"minutes": 25}}', c)[0]
    assert grade('{"function": {"name": "set_timer", "arguments": "{\\"minutes\\": 25}"}}', c)[0]
    assert not grade('{"name": "get_weather", "arguments": {}}', c)[0]


def test_contains_detects_leaks():
    c = {"type": "contains", "none": ["Sarah"], "all": ["[EMAIL]"]}
    assert grade("Hi, I'm [NAME], email [EMAIL]", c)[0]
    assert not grade("Hi, I'm Sarah, email [EMAIL]", c)[0]


def test_infer_check():
    assert infer_check({"a": 1})["type"] == "json_fields"
    assert infer_check(42)["type"] == "number"
    assert infer_check("billing", ["billing", "bug"])["type"] == "label"
    assert infer_check("A much longer free text answer that is a sentence.")["type"] == "similar"
    assert token_f1("the cat sat", "the cat sat") == 1.0


def test_unknown_check_type_raises():
    import pytest
    with pytest.raises(ValueError):
        grade("x", {"type": "nope"})
