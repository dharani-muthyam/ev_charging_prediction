"""Leakage-safe demand lag features for one-hour-ahead prediction.

Prediction task
---------------
For every zone ``z`` and timestamp ``t`` we predict the demand observed *at*
``t`` from demand observed strictly *before* ``t``::

    target      = demand_kwh[z, t]
    predictors  = demand_kwh[z, t-1]   (lag_1)
                  demand_kwh[z, t-24]  (lag_24)

Why those two lags line up with a 1-hour-ahead horizon
------------------------------------------------------
* ``lag_1``  = demand one hour before the target. It is the most recent
  observation available at prediction time (t-1) and therefore the single most
  informative predictor of ``t``.
* ``lag_24`` = demand at the same hour on the previous day. It encodes the
  daily (circadian) cycle, which dominates EV charging demand, and it also sits
  strictly in the past relative to ``t``.

Both are *shifted backwards*, so every predictor value is known at or before
``t-1``. No value from ``t`` or later is ever used as an input — that is the
definition of a leak-free 1-hour-ahead setup.

Leakage controls (deliberate)
-----------------------------
1. Lags are computed with ``groupby("zone_id")[col].shift(n)`` on the
   chronologically ordered history **before** any train/test row selection, so
   the split cannot change the lag values and no future row can leak in.
2. The shift is confined to a zone group; a lag never crosses a zone boundary.
   The first rows of each zone therefore have NaN lags by construction, and
   ``pandas.shift`` is the authoritative check that zone isolation holds
   (a global ``shift`` would wrongly fill zone A's first row from zone B).
3. ``demand_kwh`` is the target and is **not** kept as a contemporaneous
   predictor. The output frame keeps ``demand_kwh`` only so the target stays
   attached to its row; downstream modelling code must treat it as the label.
   A guard rejects any frame whose predictor set contains ``demand_kwh``.

Handling of insufficient history
--------------------------------
Rows whose lags are not both present are **dropped**, never imputed. That is
the honest choice for a regression target that is itself observed: filling a
lag would inject a value the model did not actually see at prediction time and
would silently bias the evaluation. Because the split is chronological, the
warm-up rows that get dropped all live at the *start* of each zone's training
window; the test window is untouched.

Usage
-----
    python -m src.features.demand_lags
    python -m src.features.demand_lags --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# The lag offsets, in hours. Kept as an ordered tuple so the feature order in
# the output is deterministic and the tests can iterate over it.
LAG_HOURS: tuple[int, ...] = (1, 24)

TARGET_COLUMN = "demand_kwh"
ZONE_COLUMN = "zone_id"
TIME_COLUMN = "time"

# Columns that must NOT be predictors — the target is one of them.
NON_PREDICTOR_COLUMNS: tuple[str, ...] = (TIME_COLUMN, ZONE_COLUMN, TARGET_COLUMN)

# Output filenames. The verified preprocessing artefacts are kept untouched;
# these are new, additive files (see the module docstring / report).
FEATURES_FILENAME = "features_lag.csv"
TRAIN_FILENAME = "features_train.csv"
TEST_FILENAME = "features_test.csv"
FEATURES_META_FILENAME = "features_summary.json"

# Exact-float CSV serialisation, matching src.preprocessing.prepare_data.
USE_REPR_FLOATS = True


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_processed(path: Path) -> pd.DataFrame:
    """Load a processed CSV, forcing ``zone_id`` to string and ``time`` to datetime.

    ``float_precision="round_trip"`` recovers the exact double written by the
    preprocessing step, so lag arithmetic starts from the identical values.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    cols = pd.read_csv(path, nrows=0).columns
    dtype_map = {ZONE_COLUMN: str} if ZONE_COLUMN in cols else None
    parse = [TIME_COLUMN] if TIME_COLUMN in cols else None
    df = pd.read_csv(
        path, dtype=dtype_map, parse_dates=parse, float_precision="round_trip"
    )
    if ZONE_COLUMN in df.columns:
        df["zone_id"] = df[ZONE_COLUMN].astype(str)
    return df


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------


def _dump_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV preserving exact float64 round-trip for float columns.

    Mirrors the preprocessing writer: floats go through Python's ``repr`` so the
    identical double comes back on reload; integer/other columns are untouched.
    """
    out = df.copy()
    if USE_REPR_FLOATS:
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda v: repr(float(v)) if pd.notna(v) else "")
    out.to_csv(path, index=False)


def lag_column_names(lag_hours: tuple[int, ...] = LAG_HOURS) -> list[str]:
    """Return the ordered lag feature names, e.g. ``["lag_1", "lag_24"]``."""
    return [f"lag_{h}" for h in lag_hours]


def add_lag_features(
    df: pd.DataFrame,
    lag_hours: tuple[int, ...] = LAG_HOURS,
    zone_column: str = ZONE_COLUMN,
    time_column: str = TIME_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Attach within-zone lag features to an already-sorted long frame.

    The frame is sorted by ``(zone_id, time)`` first (stable sort) and the lags
    are then produced with a per-zone ``shift``. Shifting inside the group is
    what guarantees zone isolation: each zone's series starts with NaN lags
    rather than borrowing the previous zone's tail.

    Returns a new frame; the input is not mutated. ``demand_kwh`` is retained
    as the target only — it is never turned into a contemporaneous predictor.
    """
    for col in (zone_column, time_column, target_column):
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col!r}")

    out = df.sort_values([zone_column, time_column], kind="mergesort").reset_index(drop=True)
    # Work on a copy so callers' frames are never mutated in place.
    out = out.copy()

    grouped = out.groupby(zone_column, sort=True)[target_column]
    for h in lag_hours:
        out[f"lag_{h}"] = grouped.shift(h)

    return out


def build_feature_frame(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    demand_filename: str = "demand_long.csv",
    lag_hours: tuple[int, ...] = LAG_HOURS,
    write: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the lag-feature dataset from the verified preprocessing output.

    Steps
    -----
    1. Load ``demand_long.csv`` (the verified, leak-free long table).
    2. Sort by ``(zone_id, time)`` and add within-zone lag features.
    3. Drop warm-up rows lacking a complete lag set — no imputation.
    4. If ``write``, persist the feature frame and split summaries.

    The raw data and the existing processed artefacts are only *read*.
    Returns ``(features_df, summary_dict)``.
    """
    source_path = processed_dir / demand_filename
    long_df = load_processed(source_path)

    features = add_lag_features(long_df, lag_hours=lag_hours)

    lag_cols = lag_column_names(lag_hours)
    total_rows = len(features)
    warmup_mask = features[lag_cols].isna().any(axis=1)
    n_warmup = int(warmup_mask.sum())
    features = features.loc[~warmup_mask].reset_index(drop=True)

    # Leakage guard: the target must not sneak in as a predictor.
    predictor_cols = [c for c in features.columns if c not in NON_PREDICTOR_COLUMNS]
    if TARGET_COLUMN in predictor_cols:
        raise AssertionError("demand_kwh leaked into the predictor set.")

    # Integrity guard: no NaN anywhere in the written feature frame.
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise AssertionError(f"Unexpected NaN in feature frame: {bad}")

    summary: dict = {
        "source": str(source_path),
        "lag_hours": list(lag_hours),
        "target_column": TARGET_COLUMN,
        "predictor_columns": predictor_cols,
        "columns": list(features.columns),
        "rows_before_dropna": total_rows,
        "rows_after_dropna": int(len(features)),
        "rows_dropped_warmup": n_warmup,
        "n_zones": int(features[ZONE_COLUMN].nunique()),
        "time_min": str(features[TIME_COLUMN].min()),
        "time_max": str(features[TIME_COLUMN].max()),
        "dropped_policy": (
            "rows lacking a complete lag set are dropped, never imputed; "
            "they occur only at the start of each zone's history"
        ),
    }

    if write:
        processed_dir.mkdir(parents=True, exist_ok=True)
        _dump_csv(features, processed_dir / FEATURES_FILENAME)
        with (processed_dir / FEATURES_META_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        summary["written"] = [
            str(processed_dir / FEATURES_FILENAME),
            str(processed_dir / FEATURES_META_FILENAME),
        ]

    return features, summary


def attach_split(
    features: pd.DataFrame,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    write: bool = True,
) -> dict:
    """Attach the existing chronological train/test labels to the feature rows.

    The split is *reused*, not recomputed: each ``(zone_id, time)`` key is
    looked up in the already-verified ``train.csv`` / ``test.csv`` so the
    chronological, per-zone boundary is preserved exactly. A feature row whose
    key is in neither file raises, rather than being silently dropped.
    """
    tr = load_processed(processed_dir / "train.csv")
    te = load_processed(processed_dir / "test.csv")

    tr_keys = set(zip(tr[ZONE_COLUMN], tr[TIME_COLUMN]))
    te_keys = set(zip(te[ZONE_COLUMN], te[TIME_COLUMN]))

    keys = list(zip(features[ZONE_COLUMN], features[TIME_COLUMN]))
    split = []
    for k in keys:
        if k in tr_keys:
            split.append("train")
        elif k in te_keys:
            split.append("test")
        else:
            raise KeyError(f"Feature row not present in train or test split: {k}")
    features = features.copy()
    features["split"] = pd.Categorical(split, categories=["train", "test"])

    # Chronological separation guard, per zone.
    tr_feat = features[features["split"] == "train"]
    te_feat = features[features["split"] == "test"]
    tr_max = tr_feat.groupby(ZONE_COLUMN)[TIME_COLUMN].max()
    te_min = te_feat.groupby(ZONE_COLUMN)[TIME_COLUMN].min()
    zones = sorted(set(tr_max.index) & set(te_min.index))
    violations = [z for z in zones if not (tr_max[z] < te_min[z])]
    if violations:
        raise AssertionError(f"Split not chronological for zones: {violations[:5]}")

    summary = {
        "train_rows": int(len(tr_feat)),
        "test_rows": int(len(te_feat)),
        "n_train_zones": int(tr_feat[ZONE_COLUMN].nunique()),
        "n_test_zones": int(te_feat[ZONE_COLUMN].nunique()),
        "train_time_min": str(tr_feat[TIME_COLUMN].min()),
        "train_time_max": str(tr_feat[TIME_COLUMN].max()),
        "test_time_min": str(te_feat[TIME_COLUMN].min()),
        "test_time_max": str(te_feat[TIME_COLUMN].max()),
        "chronological_separation_ok": bool(len(violations) == 0),
        "columns": list(features.columns),
    }

    if write:
        _dump_csv(tr_feat.reset_index(drop=True), processed_dir / TRAIN_FILENAME)
        _dump_csv(te_feat.reset_index(drop=True), processed_dir / TEST_FILENAME)
        summary["written"] = [
            str(processed_dir / TRAIN_FILENAME),
            str(processed_dir / TEST_FILENAME),
        ]

    return summary


def prepare(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    write: bool = True,
) -> dict:
    """Run the full feature-engineering step and return a combined summary."""
    features, feat_summary = build_feature_frame(processed_dir, write=write)
    split_summary = attach_split(features, processed_dir, write=write)

    combined = {
        "features": feat_summary,
        "split": split_summary,
        "leakage_controls": [
            "lags computed on chronologically sorted history before row selection",
            "shift applied within each zone_id group (no cross-zone bleed)",
            "demand_kwh kept as target only, excluded from predictors",
            "warm-up rows with missing lags dropped, not imputed",
            "existing chronological train/test split reused unchanged",
        ],
    }
    return combined


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build leakage-safe demand lag features.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--no-write", action="store_true", help="dry run; write nothing")
    args = parser.parse_args(argv)

    try:
        summary = prepare(args.processed_dir, write=not args.no_write)
    except (FileNotFoundError, KeyError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
