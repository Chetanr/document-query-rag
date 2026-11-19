import pytest

from src.utils import toxicity


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.49, False), (0.5, True), (0.9, True)],
)
def test_check_toxicity_uses_toxicity_score(
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    expected: bool,
):
    class FakeModel:
        def predict(self, text: str) -> dict[str, float]:
            assert text == "sample text"
            return {"toxicity": score}

    monkeypatch.setattr(toxicity, "_load_model", lambda: FakeModel())

    assert toxicity.check_toxicity("sample text") is expected
