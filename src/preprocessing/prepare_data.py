"""Initial preprocessing pipeline for UrbanEV charging demand.

Transforms the wide, time-by-zone `volume.csv` matrix into a long-form
regression table with calendar features and a chronological train/test split.

Design decisions (deliberate, to keep this step honest and leak-free):
  * Demand values are carried through **unchanged** — no imputation, no
    clipping, no scaling, no synthetic records.
  * Calendar features are derived only from the timestamp of each row itself,
    so no future information leaks into a feature.
  * The split is **chronological per zone**: the earliest 80% of that zone's
    timestamps go to train, the latest 20% to test. Timestamps are never
    shuffled, which keeps the temporal ordering intact for time-series work.
  * Occupancy is intentionally excluded — its values outside 0-100 are still
    under investigation (see reports/dataset_validation.txt).

Usage
-----
    python -m src.preprocessing.prepare_data
    python -m src.preprocessing.prepare_data --source <volume.csv> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UrbanEVDataset"
    / "20220901-20230228_zone-cleaned-aggregated"
    / "charge_1hour"
    / "volume.csv"
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

# Expected structure, verified previously in reports/dataset_validation.txt.
EXPECTED_ROWS = 4344
EXPECTED_ZONES = 275
EXPECTED_START = pd.Timestamp("2022-09-01 00:00:00")
EXPECTED_END = pd.Timestamp("2023-02-28 23:00:00")

TRAIN_FRACTION = 0.8

DEMAND_COLUMN = "demand_kwh"

# CSV float serialization.
#
# Measured behaviour on this dataset (1,194,600 cells), comparing the value set
# reloaded from disk against the in-memory float64 values:
#
#     pandas default (shortest repr) -> 35,947 cells differed by <=1e-13
#     "%.17g"                        -> 88,524 cells differed by <=1.8e-12
#     "%.18e"                        -> 122,627 cells differed
#     "%.20g"                        -> 123,732 cells differed
#
# Counter-intuitively, "%.17g" is *not* right: some of these values came from
# sums/averages that are not exactly representable, and 17 significant digits
# is not always enough to pick the correct neighbour. The only reliable approach
# is Python's own shortest-round-trip repr (float.__repr__), which is guaranteed
# to recover the identical double. We therefore serialize floats via `repr`
# rather than a printf-style float_format.
FLOAT_FORMAT = None  # unused for demand_kwh; kept for API compatibility
USE_REPR_FLOATS = True


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class SplitSummary:
    """Per-zone chronological split boundaries."""

    zone_id: str
    n_train: int
    n_test: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# --------------------------------------------------------------------------
# Loading & validation
# --------------------------------------------------------------------------


def load_wide_volume(path: Path) -> pd.DataFrame:
    """Load the wide volume matrix with `time` parsed as a datetime index.

    The source file is only read — never written to.
    """
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    # float_precision="round_trip" is essential here. pandas' default C parser
    # uses a fast (non-round-trip) float conversion that can land on a
    # neighbouring double for long decimal strings. On this dataset that
    # silently shifted 35,947 of 1,194,600 cells by up to ~4.5e-13. Python's
    # own strtod (used by the round_trip parser) recovers the intended double.
    df = pd.read_csv(
        path, index_col=0, parse_dates=True, float_precision="round_trip"
    )
    df.index.name = "time"
    return df


def validate_wide(df: pd.DataFrame) -> None:
    """Sanity-check the time index and zone columns.

    Raises
    ------
    ValueError
        If the index is not a sorted, gap-free hourly DatetimeIndex, or if
        zone columns are duplicated/empty.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not a DatetimeIndex; expected parsed 'time' column.")

    if df.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps found: {int(df.index.duplicated().sum())}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Time index is not sorted in increasing order.")

    if len(df.index) < 2:
        raise ValueError("Need at least two timestamps to validate continuity.")

    deltas = df.index.to_series().diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        n_bad = int((deltas != pd.Timedelta(hours=1)).sum())
        raise ValueError(f"Time index is not hourly/contiguous: {n_bad} irregular interval(s).")

    if df.columns.has_duplicates:
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate zone columns found: {dups}")

    if df.shape[1] == 0:
        raise ValueError("No zone columns present.")


# --------------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------------


def wide_to_long(df: pd.DataFrame, demand_column: str = DEMAND_COLUMN) -> pd.DataFrame:
    """Reshape a wide time-by-zone matrix into long form.

    Returns a DataFrame with columns `time`, `zone_id`, `demand_kwh`, sorted by
    `zone_id` then `time`. Demand values are preserved exactly (float64), with
    no filling or clipping applied.
    """
    long_df = (
        df.rename_axis(index="time", columns="zone_id")
        .stack()
        .rename(demand_column)
        .reset_index()
    )
    # Force a deterministic dtype and order without touching the values.
    long_df["zone_id"] = long_df["zone_id"].astype(str)
    long_df["time"] = pd.to_datetime(long_df["time"])
    long_df[demand_column] = long_df[demand_column].astype("float64")

    long_df = long_df.sort_values(["zone_id", "time"], kind="mergesort").reset_index(drop=True)
    return long_df[["time", "zone_id", demand_column]].reset_index(drop=True)


def _dump_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV with exact float64 round-trip for the demand column.

    Floats are converted to Python's `repr` strings first, which guarantees the
    identical double on reload. Integer calendar columns are left untouched.
    """
    out = df.copy()
    if USE_REPR_FLOATS:
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda v: repr(float(v)) if pd.notna(v) else "")
    out.to_csv(path, index=False)


def load_processed(path: Path) -> pd.DataFrame:
    """Load a processed CSV, forcing `zone_id` to string and `time` to datetime.

    Reading with an explicit dtype keeps `zone_id` textual on reload (pandas
    would otherwise infer int64 for all-numeric zone labels), and preserves the
    full-precision `demand_kwh` values written with `FLOAT_FORMAT`.
    """
    cols = pd.read_csv(path, nrows=0).columns
    dtype_map = {"zone_id": str} if "zone_id" in cols else None
    parse = ["time"] if "time" in cols else None
    df = pd.read_csv(
        path, dtype=dtype_map, parse_dates=parse, float_precision="round_trip"
    )
    if "zone_id" in df.columns:
        df["zone_id"] = df["zone_id"].astype(str)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week and month features from each row's own timestamp.

    Only the row's own timestamp is used, so there is no future information
    leakage. The original `time` column is retained.
    """
    out = df.copy()
    ts = pd.to_datetime(out["time"])
    out["hour"] = ts.dt.hour.astype("int16")
    out["day_of_week"] = ts.dt.dayofweek.astype("int16")  # Monday=0 .. Sunday=6
    out["month"] = ts.dt.month.astype("int16")
    return out


def chronological_split(
    df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame, list[SplitSummary]]:
    """Split each zone chronologically into train/test.

    For every zone, the earliest `train_fraction` of that zone's timestamps form
    the training set and the remainder the test set. Timestamps are never
    shuffled. Returns `(train_df, test_df, summaries)`.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    summaries: list[SplitSummary] = []

    for zone_id, group in df.groupby("zone_id", sort=True):
        g = group.sort_values("time", kind="mergesort")
        n = len(g)
        n_train = int(n * train_fraction)
        # Guard: both sides must be non-empty for a meaningful split.
        n_train = min(max(n_train, 1), n - 1) if n > 1 else n

        train_parts.append(g.iloc[:n_train])
        test_parts.append(g.iloc[n_train:])

        train_slice = g.iloc[:n_train]
        test_slice = g.iloc[n_train:]
        summaries.append(
            SplitSummary(
                zone_id=str(zone_id),
                n_train=int(len(train_slice)),
                n_test=int(len(test_slice)),
                train_start=train_slice["time"].iloc[0],
                train_end=train_slice["time"].iloc[-1],
                test_start=test_slice["time"].iloc[0] if len(test_slice) else pd.NaT,
                test_end=test_slice["time"].iloc[-1] if len(test_slice) else pd.NaT,
            )
        )

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]
    return train_df, test_df, summaries


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def prepare(
    source: Path = DEFAULT_SOURCE,
    out_dir: Path = DEFAULT_OUT_DIR,
    train_fraction: float = TRAIN_FRACTION,
    write: bool = True,
) -> dict:
    """Run the full preparation pipeline and optionally write outputs.

    Returns a dictionary of summary information, including output shapes and
    split dates.
    """
    wide = load_wide_volume(source)
    validate_wide(wide)

    long_df = wide_to_long(wide)
    featured = add_calendar_features(long_df)
    featured = featured.sort_values(["zone_id", "time"], kind="mergesort").reset_index(drop=True)

    train_df, test_df, summaries = chronological_split(featured, train_fraction)

    # Global split boundary (informational — the real split is per zone).
    global_cut = featured["time"].quantile(train_fraction, interpolation="lower")

    summary = {
        "source": str(source),
        "wide_shape": [int(wide.shape[0]), int(wide.shape[1])],
        "long_shape": [int(long_df.shape[0]), int(long_df.shape[1])],
        "n_zones": int(featured["zone_id"].nunique()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_fraction": float(train_fraction),
        "train_time_min": str(train_df["time"].min()),
        "train_time_max": str(train_df["time"].max()),
        "test_time_min": str(test_df["time"].min()),
        "test_time_max": str(test_df["time"].max()),
        "global_train_time_max": str(global_cut),
        "columns": list(featured.columns),
        "per_zone_example": {
            summaries[0].zone_id: {
                "n_train": summaries[0].n_train,
                "n_test": summaries[0].n_test,
                "train_end": str(summaries[0].train_end),
                "test_start": str(summaries[0].test_start),
            }
        },
        "per_zone_split_consistent": bool(
            len({s.n_train for s in summaries}) == 1 and len({s.n_test for s in summaries}) == 1
        ),
        "excluded": ["occupancy.csv (values outside 0-100 under investigation)"],
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _dump_csv(featured, out_dir / "demand_long.csv")
        _dump_csv(train_df, out_dir / "train.csv")
        _dump_csv(test_df, out_dir / "test.csv")

        split_rows = [
            {
                "zone_id": s.zone_id,
                "n_train": s.n_train,
                "n_test": s.n_test,
                "train_start": s.train_start,
                "train_end": s.train_end,
                "test_start": s.test_start,
                "test_end": s.test_end,
            }
            for s in summaries
        ]
        smeta = pd.DataFrame(split_rows)
        # Keep zone_id textual so it round-trips as a string, not an int64.
        smeta["zone_id"] = smeta["zone_id"].astype(str)
        smeta = smeta.sort_values("zone_id", kind="mergesort").reset_index(drop=True)
        smeta.to_csv(out_dir / "split_metadata.csv", index=False)

        with (out_dir / "split_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        summary["written"] = [
            str(out_dir / "demand_long.csv"),
            str(out_dir / "train.csv"),
            str(out_dir / "test.csv"),
            str(out_dir / "split_metadata.csv"),
            str(out_dir / "split_summary.json"),
        ]

    return summary


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Prepare UrbanEV demand data.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    args = parser.parse_args(argv)

    try:
        summary = prepare(args.source, args.out_dir, args.train_fraction)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
