"""Experiment 1 — Off-the-shelf evaluation on CommonLID (spec §3)."""

from __future__ import annotations

from pathlib import Path


def run(config_path: Path) -> None:
    """Thin orchestrator: load CommonLID, sample, run each model, log to W&B."""
    raise NotImplementedError("exp1_offshelf_eval.run not implemented yet")


if __name__ == "__main__":
    import typer

    typer.run(run)
