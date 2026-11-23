from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
if load_dotenv(ENV_FILE):
    logger.debug("Loaded environment variables from %s", ENV_FILE)
else:
    logger.debug("No .env file loaded from %s; using process environment", ENV_FILE)


class SettingsError(ValueError):
    pass


def _fail(message: str) -> SettingsError:
    logger.error("Configuration error: %s", message)
    return SettingsError(message)


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    anthropic_model: str | None
    top_k: int
    toxicity_threshold: float

    @classmethod
    def from_env(cls) -> Settings:
        top_k_value = os.environ.get("TOP_K")
        if top_k_value is None:
            raise _fail("Set TOP_K in the .env file")
        try:
            top_k = int(top_k_value)
        except ValueError as error:
            raise _fail("TOP_K must be a positive integer") from error
        if top_k <= 0:
            raise _fail("TOP_K must be a positive integer")

        threshold_value = os.environ.get("TOXICITY_THRESHOLD")
        if threshold_value is None:
            raise _fail("Set TOXICITY_THRESHOLD in the .env file")
        try:
            toxicity_threshold = float(threshold_value)
        except ValueError as error:
            raise _fail(
                "TOXICITY_THRESHOLD must be a number between 0 and 1"
            ) from error
        if not 0 <= toxicity_threshold <= 1:
            raise _fail("TOXICITY_THRESHOLD must be a number between 0 and 1")

        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        anthropic_model = os.environ.get("ANTHROPIC_MODEL")
        if not anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY is not set")
        if not anthropic_model:
            logger.warning("ANTHROPIC_MODEL is not set")

        logger.debug(
            "Settings loaded: model=%s, top_k=%d, toxicity_threshold=%s",
            anthropic_model,
            top_k,
            toxicity_threshold,
        )
        return cls(
            anthropic_api_key=anthropic_api_key,
            anthropic_model=anthropic_model,
            top_k=top_k,
            toxicity_threshold=toxicity_threshold,
        )
