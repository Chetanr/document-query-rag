from types import SimpleNamespace

import pytest

from src import api
from src.config import Settings
from src.utils import toxicity as real_toxicity


class ToxicityStub:
    """Stands in for src.utils.toxicity.check_toxicity; records calls."""

    def __init__(self) -> None:
        self.result = False
        self.calls: list[tuple[str, float]] = []

    def __call__(self, text: str, threshold: float) -> bool:
        self.calls.append((text, threshold))
        return self.result


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        anthropic_model="test-model",
        top_k=3,
        toxicity_threshold=0.5,
    )


@pytest.fixture(autouse=True)
def toxicity_stub(monkeypatch: pytest.MonkeyPatch) -> ToxicityStub:
    """Never load the real Detoxify model in tests."""

    stub = ToxicityStub()
    # Replace the module as seen by api.py only; the real module stays intact
    # so tests/test_toxicity.py can still exercise it with a fake model.
    monkeypatch.setattr(api, "toxicity", SimpleNamespace(check_toxicity=stub))
    return stub


@pytest.fixture(autouse=True)
def forbid_real_toxicity_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_model_loaded():
        raise AssertionError("Tests must not load the real Detoxify model")

    monkeypatch.setattr(real_toxicity, "_load_model", fail_if_model_loaded)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    """API endpoints get fixed settings instead of reading .env / os.environ."""
    monkeypatch.setitem(
        api.app.dependency_overrides, api.get_settings, lambda: test_settings
    )
