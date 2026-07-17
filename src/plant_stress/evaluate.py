"""Regression metrics and threshold-based stress classification reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .config import STRESS_ORDER, drought_level, heat_level


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R^2 for one target."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    error = y_pred - y_true
    variance = np.var(y_true)
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float("nan") if variance < 1e-12 else float(1.0 - np.mean(error**2) / variance),
    }


def stress_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, kind: str
) -> tuple[pd.DataFrame, str]:
    """Confusion matrix + per-class report after binning values into bands.

    Args:
        kind: 'heat' (bins Tleaf) or 'drought' (bins gsw).

    Only the bands actually present are shown. On a 16-row validation split most
    of the five bands are empty, and printing them as zero rows implies coverage
    the data does not have.
    """
    binner = {"heat": heat_level, "drought": drought_level}.get(kind)
    if binner is None:
        raise ValueError("kind must be 'heat' or 'drought'")

    true_labels = [binner(float(v)) for v in y_true]
    pred_labels = [binner(float(v)) for v in y_pred]

    present = [b for b in STRESS_ORDER if b in set(true_labels) | set(pred_labels)]
    matrix = confusion_matrix(true_labels, pred_labels, labels=present)
    table = pd.DataFrame(
        matrix,
        index=pd.Index(present, name="true"),
        columns=pd.Index(present, name="predicted"),
    )
    report = classification_report(
        true_labels, pred_labels, labels=present, zero_division=0
    )
    return table, report


def summarize(y_true: np.ndarray, y_pred: np.ndarray, target: str, kind: str) -> None:
    """Print the regression + classification summary for one target."""
    metrics = regression_metrics(y_true, y_pred)
    unit = "deg C" if target.lower() == "tleaf" else "mol m-2 s-1"

    print(f"\n=== {target} (n={metrics['n']}) ===")
    print(f"  MAE  = {metrics['mae']:.4f} {unit}")
    print(f"  RMSE = {metrics['rmse']:.4f} {unit}")
    print(f"  R2   = {metrics['r2']:.4f}")

    table, report = stress_confusion(y_true, y_pred, kind)
    print(f"\n  {kind} confusion matrix:\n{table.to_string()}")
    print(f"\n{report}")
