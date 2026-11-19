from functools import cache

from detoxify import Detoxify


@cache
def _load_model() -> Detoxify:
    return Detoxify("original")


def check_toxicity(text: str, threshold: float) -> bool:
    toxicity_score = _load_model().predict(text)["toxicity"]
    return toxicity_score >= threshold

