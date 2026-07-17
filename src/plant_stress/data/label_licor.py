"""Turn raw LI-600 porometer exports into a labelled ground-truth table.

The LI-600 writes one CSV per session with a metadata preamble. This module
concatenates the sessions, coerces the physiological channels to numbers, and
derives the categorical stress labels used for classification reporting.

Ground-truth channels we care about:
    gsw     stomatal conductance    mol m-2 s-1
    Tleaf   leaf temperature        deg C
    Fv/Fm   max PSII quantum yield  ~0.83 healthy, <0.75 stressed
    PhiPS2  operating PSII yield
    ETR     electron transport rate

Only gsw and Tleaf are used as regression targets. The fluorescence channels
are retained for the descriptive statistics in the thesis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import drought_level, heat_level

NUMERIC_COLUMNS = ["gsw", "Tleaf", "Fv/Fm", "PhiPS2", "ETR", "VPDleaf", "E_apparent"]


def load_licor_exports(paths: list[Path]) -> pd.DataFrame:
    """Concatenate LI-600 CSV exports into one frame."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("No LI-600 CSV files given.")
    return pd.concat(frames, ignore_index=True)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Force physiological channels to numeric, turning junk into NaN.

    NaNs are left in place on purpose. Mean-imputing a physiological reading
    invents a measurement that was never taken; downstream code drops those
    rows instead.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_stress_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach categorical drought/heat bands derived from gsw and Tleaf."""
    df = df.copy()
    if "gsw" in df.columns:
        df["drought_stress"] = df["gsw"].apply(
            lambda v: "Unknown" if pd.isna(v) else drought_level(float(v))
        )
    if "Tleaf" in df.columns:
        df["heat_stress"] = df["Tleaf"].apply(
            lambda v: "Unknown" if pd.isna(v) else heat_level(float(v))
        )
    return df


def build(paths: list[Path]) -> pd.DataFrame:
    """Full pipeline: load -> coerce -> label."""
    return add_stress_labels(coerce_numeric(load_licor_exports(paths)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="LI-600 CSV exports")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    df = build(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {args.output}  shape={df.shape}")
    for col in ("drought_stress", "heat_stress"):
        if col in df.columns:
            print(f"\n{col}:\n{df[col].value_counts().to_string()}")


if __name__ == "__main__":
    main()
