import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.modexai during tests."""
    monkeypatch.setenv("MODEXAI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MODEXAI_REGISTRY", raising=False)
    yield tmp_path / "home"
