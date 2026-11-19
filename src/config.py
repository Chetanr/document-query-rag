import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: Optional[str]
    anthropic_model: Optional[str]
    top_k: int
    toxicity_threshold: float

    @classmethod
    def from_env(cls) -> "Settings":
        top_k_value = os.environ.get("TOP_K")
        if top_k_value is None:
            raise SettingsError("Set TOP_K in the .env file")
        try:
            top_k = int(top_k_value)
        except ValueError as error:
            raise SettingsError("TOP_K must be a positive integer") from error
        if top_k <= 0:
            raise SettingsError("TOP_K must be a positive integer")

        threshold_value = os.environ.get("TOXICITY_THRESHOLD")
        if threshold_value is None:
            raise SettingsError("Set TOXICITY_THRESHOLD in the .env file")
        try:
            toxicity_threshold = float(threshold_value)
        except ValueError as error:
            raise SettingsError("TOXICITY_THRESHOLD must be a number between 0 and 1") from error
        if not 0 <= toxicity_threshold <= 1:
            raise SettingsError("TOXICITY_THRESHOLD must be a number between 0 and 1")

        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL"),
            top_k=top_k,
            toxicity_threshold=toxicity_threshold,
        )

