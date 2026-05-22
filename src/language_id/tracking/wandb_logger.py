"""W&B logging: consistent run naming, artifacts, sklearn plots (spec §5, §12.2)."""

from __future__ import annotations

from typing import Any


def init_run(
    experiment: str,
    model: str,
    dataset: str,
    seed: int,
    config: dict[str, Any],
) -> Any:
    """Initialize a W&B run with the project / name convention from spec §12.2.

    Project: `lid-bench-<experiment>` (e.g. `lid-bench-exp1`).
    Name: `{experiment}-{model}-{dataset}-{seed}`.
    """
    raise NotImplementedError("init_run not implemented yet (spec §12.2)")


def log_predictions(predictions: Any) -> None:
    raise NotImplementedError("log_predictions not implemented yet")


def log_unparseable(model: str, raws: list[str], gold: list[str]) -> None:
    """Log unparseable LLM responses to a dedicated W&B table (spec §8)."""
    raise NotImplementedError("log_unparseable not implemented yet")
