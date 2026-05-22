"""Experiment 2 — Train LogReg, NGram-NB, and XLM-R LoRA on CommonVoiceLID (spec §3)."""

from __future__ import annotations

from pathlib import Path


def run(config_path: Path) -> None:
    """Thin orchestrator: load CommonVoiceLID, train each classifier, log to W&B."""
    raise NotImplementedError("exp2_train_classifiers.run not implemented yet")


if __name__ == "__main__":
    import typer

    typer.run(run)
