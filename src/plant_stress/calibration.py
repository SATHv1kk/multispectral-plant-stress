"""Ridge calibration of raw Tleaf predictions.

The network learns the *ranking* of leaf temperature well but its raw output is
slightly compressed toward the training mean -- the usual regression-to-the-mean
of a small-sample regressor. A cheap ridge model on top corrects the scale.

Features: [pred, pred^2, ndvi, ndre]. The quadratic term lets the correction
bend rather than merely rescale; NDVI/NDRE let it vary with canopy state.

LEAKAGE RULES -- the reason `tleaf_calibrator_NOLEAK.pkl` carries that suffix:

  * The calibrator is fit on the TRAINING split only, then applied unchanged to
    validation. Fitting on validation rows would be fitting on the answer.
  * A per-date calibrator is used only for dates that appear in TRAINING and
    have enough samples to support one; every other date falls back to the
    global calibrator. A per-date calibrator fitted on a validation date would
    leak that date's label distribution.
  * VPDleaf is deliberately NOT a feature. It is computed by the LI-600 FROM
    Tleaf, so including it leaks the target and produces the R^2~=0.9955 figure
    reported in the thesis as a diagnostic only, never as a headline result.
    Future work replaces it with ambient VPD, which is leakage-free.

Minimum samples for a per-date fit is 6: with 4 features plus an intercept,
fewer rows can interpolate the training points exactly and generalise to
nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURE_BASE = ["ndvi", "ndre"]
MIN_SAMPLES_PER_DATE = 6
DEFAULT_ALPHA = 1.0


def build_features(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """[pred, pred^2, ndvi, ndre]; missing index columns are simply omitted."""
    out = pd.DataFrame(index=df.index)
    out[pred_col] = df[pred_col].astype(float)
    out[f"{pred_col}_sq"] = out[pred_col] ** 2
    for col in FEATURE_BASE:
        if col in df.columns:
            out[col] = df[col].astype(float)
    return out


class RidgeCalibrator:
    """Standardised ridge regression mapping raw prediction -> calibrated value."""

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.columns: list[str] = []
        self.pred_col: str = ""

    def fit(self, df: pd.DataFrame, pred_col: str, target_col: str) -> "RidgeCalibrator":
        features = build_features(df, pred_col).fillna(0.0)
        self.columns = features.columns.tolist()
        self.pred_col = pred_col
        self.model.fit(self.scaler.fit_transform(features.values), df[target_col].astype(float).values)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features = build_features(df, self.pred_col).reindex(columns=self.columns).fillna(0.0)
        return self.model.predict(self.scaler.transform(features.values))


class DateAwareCalibrator:
    """Per-date calibrators where the data supports them, global fallback elsewhere."""

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        self.global_calibrator: RidgeCalibrator | None = None
        self.per_date: dict[str, RidgeCalibrator] = {}

    def fit(self, train: pd.DataFrame, pred_col: str, target_col: str, date_col: str = "date"):
        self.global_calibrator = RidgeCalibrator(self.alpha).fit(train, pred_col, target_col)
        self.per_date = {
            date: RidgeCalibrator(self.alpha).fit(group, pred_col, target_col)
            for date, group in train.groupby(date_col)
            if len(group) >= MIN_SAMPLES_PER_DATE
        }
        print(
            f"[calibration] global + {len(self.per_date)} per-date calibrators "
            f"(dates with >= {MIN_SAMPLES_PER_DATE} training rows)"
        )
        return self

    def predict(self, df: pd.DataFrame, date_col: str = "date"):
        """Returns (calibrated_values, scope_used_per_row)."""
        if self.global_calibrator is None:
            raise RuntimeError("Call fit() before predict().")

        values, scopes = np.empty(len(df), dtype=float), []
        for i, (_, row) in enumerate(df.iterrows()):
            date = row[date_col]
            frame = row.to_frame().T
            if date in self.per_date:
                values[i] = self.per_date[date].predict(frame)[0]
                scopes.append(f"per-date:{date}")
            else:
                values[i] = self.global_calibrator.predict(frame)[0]
                scopes.append("global")
        return values, scopes


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 that returns NaN instead of exploding when the target is constant."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    variance = np.var(y_true)
    if variance < 1e-12:
        return float("nan")
    return float(1.0 - np.mean((y_true - y_pred) ** 2) / variance)
