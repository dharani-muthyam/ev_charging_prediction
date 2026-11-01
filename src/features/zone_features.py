"""Static, leakage-free zone features from UrbanEV zone/station metadata.

Why these features are safe
---------------------------
``zone-information.csv`` and ``station_information.csv`` are **static**
administrative snapshots: a zone's pile count, area and perimeter do not change
over the six-month study window, and they are known when a forecast is
requested. They therefore carry no future information and cannot leak the
target (``demand_kwh``).

What is added
-------------
Joined one-to-one onto the existing ``(time, zone_id)`` feature rows:

* ``zone_charge_count``   — total charging piles in the zone (from zone-information)
* ``zone_area_km2``       — zone area, converted m² → km² (raw values are in m²)
* ``zone_perimeter_km``   — zone perimeter, converted m → km
* ``zone_piles_per_km2``  — charge_count / area_km² (infrastructure density)
* ``zone_n_stations``     — number of filtered stations in the zone (station-information)
* ``zone_longitude``, ``zone_latitude`` — zone centroid coordinates (WGS-84)

Explicitly **not** added:

* ``TAZID`` / ``station_id`` used as a *numeric* predictor — a zone ID is a
  label, not a quantity, so its numeric ordering is meaningless.
* Any per-timestamp quantity from ``occupancy.csv`` or ``duration.csv``: both
  are *contemporaneous* observations of the same hour being predicted and
  would leak the target (see ``docs/data_quality.md``).

Join-safety checks (raised, never silently repaired)
----------------------------------------------------
1. Exactly one row per key on both sides — no many-to-many fan-out.
2. The output row count must equal the input row count.
3. Every zone in the features must match a ``zone-information`` row.
4. No missing values introduced by the join.
5. ``station_information.charge_count`` summed per zone is reported against
   ``zone-information.charge_count`` so any divergence is visible, not hidden.

Usage
-----
    python -m src.features.zone_features
    python -m src.features.zone_features --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ZONE_INFO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UrbanEVDataset"
    / "20220901-20230228_zone-cleaned-aggregated"
    / "zone-information.csv"
)
DEFAULT_STATION_INFO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UrbanEVDataset"
    / "20220901-20230228_zone-cleaned-aggregated"
    / "station_information.csv"
)

ZONE_COLUMN = "zone_id"
TIME_COLUMN = "time"

FEATURE_COLUMNS: tuple[str, ...] = (
    "zone_charge_count",
    "zone_area_km2",
    "zone_perimeter_km",
    "zone_piles_per_km2",
    "zone_n_stations",
    "zone_longitude",
    "zone_latitude",
)

TRAIN_IN = "features_train.csv"
TEST_IN = "features_test.csv"
TRAIN_OUT = "features_train_zone.csv"
TEST_OUT = "features_test_zone.csv"
SUMMARY_FILENAME = "zone_features_summary.json"

# Exact-float CSV serialisation, matching the rest of the pipeline.
USE_REPR_FLOATS = True


# --------------------------------------------------------------------------
# Metadata loading
# --------------------------------------------------------------------------


def _read(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} metadata not found: {path}")
    return pd.read_csv(path, float_precision="round_trip")


def load_zone_table(
    zone_info_path: Path = DEFAULT_ZONE_INFO,
    station_info_path: Path = DEFAULT_STATION_INFO,
) -> pd.DataFrame:
    """Build one row per ``zone_id`` from the static zone and station tables.

    ``zone_id`` is returned as a **string** to match the demand table (both
    source files store ``TAZID`` as int64, which would otherwise silently fail
    to match text keys).
    """
    zones = _read(zone_info_path, "zone-information")
    stations = _read(station_info_path, "station-information")

    for df, name, key in (
        (zones, "zone-information", "TAZID"),
        (stations, "station-information", "TAZID"),
    ):
        if df[key].isna().any():
            raise ValueError(f"{name}: missing {key} value(s)")
        if df[key].duplicated().any() and name == "zone-information":
            raise ValueError(f"{name}: duplicate {key} — expected one row per zone")

    zones = zones.assign(TAZID=zones["TAZID"].astype(str))
    stations = stations.assign(TAZID=stations["TAZID"].astype(str))

    # Aggregate stations to zones *before* joining, so the join is one-to-one
    # by construction (a direct station→demand join would fan out).
    station_agg = (
        stations.groupby("TAZID", as_index=False)
        .agg(zone_n_stations=("station_id", "nunique"))
        .rename(columns={"TAZID": ZONE_COLUMN})
    )
    if station_agg[ZONE_COLUMN].duplicated().any():
        raise ValueError("station aggregation produced duplicate zones")

    zone_table = pd.DataFrame(
        {
            ZONE_COLUMN: zones["TAZID"],
            "zone_charge_count": zones["charge_count"].astype("int32"),
            "zone_area_km2": zones["area"].astype(float) / 1_000_000.0,
            "zone_perimeter_km": zones["perimeter"].astype(float) / 1_000.0,
            "zone_longitude": zones["longitude"].astype(float),
            "zone_latitude": zones["latitude"].astype(float),
        }
    )
    # Vectorised division; a zero-area zone would yield inf, so guard it.
    area = zone_table["zone_area_km2"].to_numpy(dtype=float)
    if not np.all(area > 0):
        raise ValueError("zone-information contains a non-positive area")
    zone_table["zone_piles_per_km2"] = zone_table["zone_charge_count"].to_numpy(
        dtype=float
    ) / area

    zone_table = zone_table.merge(station_agg, on=ZONE_COLUMN, how="left", validate="one_to_one")

    if zone_table[ZONE_COLUMN].duplicated().any():
        raise AssertionError("zone table must have exactly one row per zone")
    if zone_table[list(FEATURE_COLUMNS)].isna().any().any():
        bad = zone_table.columns[zone_table[list(FEATURE_COLUMNS)].isna().any()].tolist()
        raise AssertionError(f"zone table has NaN in {bad}")
    return zone_table


# --------------------------------------------------------------------------
# Join
# --------------------------------------------------------------------------


def attach_zone_features(
    features: pd.DataFrame,
    zone_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Left-join the static zone attributes onto ``(time, zone_id)`` rows.

    Row order and row count are preserved exactly. Returns ``(out, report)``.
    """
    if zone_table is None:
        zone_table = load_zone_table()

    if ZONE_COLUMN not in features.columns:
        raise KeyError(f"{ZONE_COLUMN!r} missing from the feature frame")

    before_rows = len(features)
    zones_in_data = set(features[ZONE_COLUMN].astype(str).unique())
    zones_in_meta = set(zone_table[ZONE_COLUMN].astype(str))

    unmatched = sorted(zones_in_data - zones_in_meta)
    if unmatched:
        raise AssertionError(
            f"{len(unmatched)} zone(s) in the demand data have no zone metadata: "
            f"{unmatched[:10]}"
        )

    left = features.assign(**{ZONE_COLUMN: features[ZONE_COLUMN].astype(str)})
    out = left.merge(
        zone_table[[ZONE_COLUMN, *FEATURE_COLUMNS]],
        on=ZONE_COLUMN,
        how="left",
        validate="many_to_one",  # each demand row matches at most one zone row
        sort=False,
    )
    if len(out) != before_rows:
        raise AssertionError(
            f"row count changed by the zone join: {before_rows} -> {len(out)} "
            "(many-to-many join suspected)"
        )
    if out[list(FEATURE_COLUMNS)].isna().any().any():
        bad = out.columns[out[list(FEATURE_COLUMNS)].isna().any()].tolist()
        raise AssertionError(f"zone join introduced NaN in {bad}")

    report = {
        "rows_before": int(before_rows),
        "rows_after": int(len(out)),
        "zones_in_demand": len(zones_in_data),
        "zones_with_metadata": len(zones_in_meta),
        "unmatched_zones": unmatched,
        "zones_in_metadata_not_in_demand": sorted(zones_in_meta - zones_in_data),
        "join_type": "many_to_one (static zone attributes)",
        "features": list(FEATURE_COLUMNS),
    }
    return out, report


def metadata_consistency(
    zone_table: pd.DataFrame,
    station_info_path: Path = DEFAULT_STATION_INFO,
) -> dict:
    """Compare station-level pile counts with the zone-level total (diagnostic)."""
    stations = _read(station_info_path, "station-information")
    per_zone = (
        stations.assign(TAZID=stations["TAZID"].astype(str))
        .groupby("TAZID")["charge_count"]
        .sum()
    )
    zone_counts = zone_table.set_index(ZONE_COLUMN)["zone_charge_count"]
    common = per_zone.index.intersection(zone_counts.index)
    diff = (per_zone.loc[common] - zone_counts.loc[common]).abs()
    return {
        "zones_compared": int(len(common)),
        "zones_where_counts_agree": int((diff == 0).sum()),
        "max_abs_difference": float(diff.max()) if len(diff) else 0.0,
        "correlation": float(
            np.corrcoef(per_zone.loc[common].to_numpy(float), zone_counts.loc[common].to_numpy(float))[0, 1]
        )
        if len(common) > 1
        else None,
        "note": (
            "station_information.csv is a *filtered* subset of stations; "
            "zone-information.charge_count is the authoritative zone total. "
            "The zone total is used as the predictor, station count only as a "
            "secondary descriptor."
        ),
    }


# --------------------------------------------------------------------------
# Pipeline entry points
# --------------------------------------------------------------------------


def _dump_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if USE_REPR_FLOATS:
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda v: repr(float(v)) if pd.notna(v) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _load(path: Path) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(
        path,
        dtype={ZONE_COLUMN: str},
        parse_dates=[TIME_COLUMN] if TIME_COLUMN in cols else None,
        float_precision="round_trip",
    )
    df[ZONE_COLUMN] = df[ZONE_COLUMN].astype(str)
    return df


def build(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    zone_info_path: Path = DEFAULT_ZONE_INFO,
    station_info_path: Path = DEFAULT_STATION_INFO,
    write: bool = True,
) -> dict:
    """Attach zone features to the existing train/test feature tables."""
    zone_table = load_zone_table(zone_info_path, station_info_path)

    train_in = processed_dir / TRAIN_IN
    test_in = processed_dir / TEST_IN
    for p in (train_in, test_in):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python -m src.features.demand_lags` first."
            )

    train = _load(train_in)
    test = _load(test_in)

    train_out, train_report = attach_zone_features(train, zone_table)
    test_out, test_report = attach_zone_features(test, zone_table)

    summary = {
        "sources": {"zone_information": str(zone_info_path), "station_information": str(station_info_path)},
        "feature_columns": list(FEATURE_COLUMNS),
        "train": train_report,
        "test": test_report,
        "zone_table_rows": int(len(zone_table)),
        "metadata_consistency": metadata_consistency(zone_table, station_info_path),
        "excluded_from_predictors": {
            "TAZID": "zone label; numeric ordering is meaningless",
            "station_id": "station label; aggregated to a zone count instead",
            "occupancy.csv": "contemporaneous observation of the predicted hour (would leak)",
            "duration.csv": "contemporaneous observation of the predicted hour (would leak)",
        },
        "leakage_notes": [
            "all zone features are static administrative attributes",
            "no timestamp-dependent quantity from the raw source is joined",
            "join validated as many_to_one; row count preserved",
        ],
    }

    if write:
        _dump_csv(train_out, processed_dir / TRAIN_OUT)
        _dump_csv(test_out, processed_dir / TEST_OUT)
        with (processed_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        summary["written"] = [
            str(processed_dir / TRAIN_OUT),
            str(processed_dir / TEST_OUT),
            str(processed_dir / SUMMARY_FILENAME),
        ]

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attach static zone features.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--zone-info", type=Path, default=DEFAULT_ZONE_INFO)
    parser.add_argument("--station-info", type=Path, default=DEFAULT_STATION_INFO)
    parser.add_argument("--no-write", action="store_true", help="dry run; write nothing")
    args = parser.parse_args(argv)

    try:
        summary = build(
            args.processed_dir, args.zone_info, args.station_info, write=not args.no_write
        )
    except (FileNotFoundError, KeyError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
