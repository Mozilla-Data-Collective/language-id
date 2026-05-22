"""Global configuration: API keys, paths, W&B project names.

Loads from environment variables (and a `.env` file if present) via pydantic-settings.
YAML configs in `configs/` carry per-model and per-experiment settings; this module
only holds the global runtime knobs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Repository layout
    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    configs_dir: Path = Field(default_factory=lambda: Path("configs"))
    results_dir: Path = Field(default_factory=lambda: Path("results"))
    cache_dir: Path = Field(default_factory=lambda: Path(".diskcache"))

    # External services
    datacollective_api_key: str | None = None
    any_llm_api_key: str | None = None
    hf_token: str | None = None
    wandb_api_key: str | None = None
    wandb_entity: str | None = None


def get_settings() -> Settings:
    return Settings()
