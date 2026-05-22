"""Calibration metrics: ECE + reliability diagrams (spec §9.2, where probabilities available)."""

from __future__ import annotations

from collections.abc import Sequence


def expected_calibration_error(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    n_bins: int = 15,
) -> float:
    raise NotImplementedError("expected_calibration_error not implemented yet")


def reliability_diagram_data(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """Return per-bin bin_center, accuracy, confidence, count — ready for matplotlib."""
    raise NotImplementedError("reliability_diagram_data not implemented yet")
