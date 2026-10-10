import json

from modexai import config
from modexai.registry import load_registry, validate_listing


def test_bundled_registry_loads_and_is_valid():
    reg = load_registry()
    assert len(reg) >= 20
    for m in reg.models:
        assert validate_listing(m.raw) == [], m.id


def test_lookup_by_id_and_tag():
    reg = load_registry()
    assert reg.get("qwen/qwen3-1.7b").ollama_tag == "qwen3:1.7b"
    assert reg.get("qwen3:1.7b").id == "qwen/qwen3-1.7b"
    assert reg.get("nuextract:latest").id == "numind/nuextract"


def test_search_prefers_specialists_and_respects_size():
    reg = load_registry()
    found = reg.search(task="prompt-injection")
    assert found[0].id == "nokast/securerag-0.5b"
    assert all(m.size_gb <= 1.0 for m in reg.search(max_size_gb=1.0))


def test_private_listings_are_merged(isolated_home):
    d = isolated_home / "listings"
    d.mkdir(parents=True)
    (d / "mine.json").write_text(json.dumps({
        "id": "acme/claims-1b", "name": "Claims 1B", "publisher": "Acme", "summary": "Internal claims triage.",
        "license": "proprietary", "tasks": ["claims"], "source": {"type": "ollama", "tag": "acme-claims:1b"}}), encoding="utf-8")
    reg = load_registry()
    assert reg.get("acme/claims-1b").raw["_private"] is True


def test_validation_catches_mistakes():
    errs = validate_listing({"id": "Bad Id", "tasks": [], "source": {"type": "ftp"}})
    assert any("id" in e for e in errs)
    assert any("tasks" in e for e in errs)
    assert any("source.type" in e for e in errs)


def test_bundled_registry_file_matches_listings():
    from modexai.build import build
    assert build(check=True) == 0
    assert config.BUNDLED_REGISTRY.exists()
