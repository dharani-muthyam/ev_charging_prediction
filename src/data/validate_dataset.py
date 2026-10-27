"""Full-dataset validation for the UrbanEV hourly zone-level charging data.

This module is strictly READ-ONLY with respect to the source dataset. It loads
each CSV, reports structural and statistical properties, and checks cross-file
consistency. It never writes back to, clips, or transforms the original files.

Usage
-----
    python -m src.data.validate_dataset
    python -m src.data.validate_dataset --data-dir <path> --out <path>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_FILES = ("volume.csv", "volume-11kw.csv", "occupancy.csv", "duration.csv")

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "UrbanEVDataset"
    / "20220901-20230228_zone-cleaned-aggregated"
    / "charge_1hour"
)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "reports" / "dataset_validation.txt"

TARGET_FILES = ("volume.csv", "volume-11kw.csv", "occupancy.csv", "duration.csv")

EXPECTED_START = pd.Timestamp("2022-09-01 00:00:00")
EXPECTED_END = pd.Timestamp("2023-02-28 23:00:00")
EXPECTED_ROWS = 4344
EXPECTED_ZONES = 275


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class FileReport:
    """Validation results for a single CSV file."""

    name: str
    exists: bool = False
    error: str | None = None
    n_rows: int = 0
    n_cols: int = 0
    time_min: pd.Timestamp | None = None
    time_max: pd.Timestamp | None = None
    duplicate_timestamps: int = 0
    duplicate_zone_columns: list[str] = field(default_factory=list)
    missing_cells: int = 0
    nonnumeric_cells: int = 0
    negative_cells: int = 0
    total_cells: int = 0
    cell_min: float | None = None
    cell_max: float | None = None
    cell_mean: float | None = None
    out_of_0_100_cells: int = 0
    out_of_0_100_examples: list[str] = field(default_factory=list)
    timestamps: pd.DatetimeIndex | None = None
    zone_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Core validation
# --------------------------------------------------------------------------


def load_wide_csv(path: Path) -> pd.DataFrame:
    """Load a wide UrbanEV CSV, parsing the first `time` column as datetime.

    The `time` column is set as the index. Original values are not altered;
    pandas parses on read, so the on-disk file is untouched.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    return df


def validate_dataframe(df: pd.DataFrame, name: str) -> FileReport:
    """Validate an already-loaded wide charging DataFrame.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex named `time` and one column per zone.
    name:
        Label used in the report.

    Returns
    -------
    FileReport
        A populated, immutable-by-convention summary. Nothing is modified.
    """
    rep = FileReport(name=name, exists=True)

    rep.n_rows = int(df.shape[0])
    rep.n_cols = int(df.shape[1])

    # --- time index ------------------------------------------------------
    rep.timestamps = df.index
    rep.time_min = df.index.min()
    rep.time_max = df.index.max()
    rep.duplicate_timestamps = int(df.index.duplicated().sum())

    # --- zone columns ----------------------------------------------------
    rep.zone_columns = [str(c) for c in df.columns]
    dupes = df.columns[df.columns.duplicated()].tolist()
    rep.duplicate_zone_columns = [str(c) for c in dupes]

    # --- cell-level checks ----------------------------------------------
    # Coerce to numeric without mutating the caller's frame. Cells that fail
    # coercion become NaN and are counted as non-numeric.
    numeric = df.apply(pd.to_numeric, errors="coerce")
    rep.total_cells = int(numeric.size)
    rep.nonnumeric_cells = int(numeric.isna().sum().sum() - df.isna().sum().sum())
    rep.missing_cells = int(df.isna().sum().sum())

    if numeric.size:
        rep.cell_min = float(np.nanmin(numeric.to_numpy()))
        rep.cell_max = float(np.nanmax(numeric.to_numpy()))
        rep.cell_mean = float(np.nanmean(numeric.to_numpy()))
    rep.negative_cells = int((numeric < 0).sum().sum())

    # --- 0-100 range diagnostic (reported, never clipped) ----------------
    outside = (numeric < 0) | (numeric > 100)
    rep.out_of_0_100_cells = int(outside.sum().sum())
    if rep.out_of_0_100_cells:
        # Collect a few (time, zone) locations for inspection.
        rows, cols = np.where(outside.to_numpy())
        for r, c in list(zip(rows, cols))[:10]:
            rep.out_of_0_100_examples.append(
                f"{df.index[r]} | zone {df.columns[c]} = {numeric.iloc[r, c]}"
            )

    return rep


def validate_file(path: Path) -> FileReport:
    """Validate a single CSV on disk. Captures errors instead of raising."""
    if not path.exists():
        return FileReport(name=path.name, exists=False, error="File not found")
    try:
        df = load_wide_csv(path)
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        return FileReport(name=path.name, exists=False, error=f"{type(exc).__name__}: {exc}")
    return validate_dataframe(df, path.name)


def check_hourly_continuity(index: pd.DatetimeIndex) -> tuple[bool, str]:
    """Verify the index is hourly and gap-free."""
    if len(index) < 2:
        return False, "Not enough timestamps to assess continuity"
    deltas = index.to_series().diff().dropna()
    expected = pd.Timedelta(hours=1)
    bad = deltas[deltas != expected]
    if bad.empty:
        return True, f"Continuous hourly series ({len(index)} timestamps, no gaps)"
    return False, (
        f"{len(bad)} irregular interval(s); first at {bad.index[0]} "
        f"with delta {bad.iloc[0]}"
    )


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def build_report(data_dir: Path, files: tuple[str, ...] = TARGET_FILES) -> str:
    """Run validation across all files and return the report as text."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("UrbanEV Dataset — Full Validation Report")
    add("=" * 78)
    add(f"Data directory : {data_dir}")
    add(f"Generated by   : src/data/validate_dataset.py (read-only)")
    add("")

    reports: list[FileReport] = []
    for fname in files:
        rep = validate_file(data_dir / fname)
        reports.append(rep)

    # ---------------- per-file sections ----------------
    for rep in reports:
        add("-" * 78)
        add(f"FILE: {rep.name}")
        add("-" * 78)
        if not rep.exists:
            add(f"  STATUS          : FAILED TO LOAD — {rep.error}")
            add("")
            continue

        add(f"  Shape           : {rep.n_rows} rows x {rep.n_cols} columns")
        add(f"  Time range      : {rep.time_min}  ->  {rep.time_max}")
        add(f"  Duplicate times : {rep.duplicate_timestamps}")
        add(f"  Duplicate zones : {len(rep.duplicate_zone_columns)}"
            + (f" {rep.duplicate_zone_columns}" if rep.duplicate_zone_columns else ""))
        add(f"  Missing cells   : {rep.missing_cells}")
        add(f"  Non-numeric     : {rep.nonnumeric_cells}")
        add(f"  Negative cells  : {rep.negative_cells}")
        add(f"  Total cells     : {rep.total_cells}")
        add(f"  Min             : {rep.cell_min:.6f}")
        add(f"  Max             : {rep.cell_max:.6f}")
        add(f"  Mean            : {rep.cell_mean:.6f}")

        ok, msg = check_hourly_continuity(rep.timestamps)
        add(f"  Hourly gap-free : {'YES' if ok else 'NO'} — {msg}")

        if rep.name == "occupancy.csv":
            pct = (rep.out_of_0_100_cells / rep.total_cells * 100) if rep.total_cells else 0.0
            add("")
            add(f"  [occupancy check] cells outside 0-100 : {rep.out_of_0_100_cells}"
                f" ({pct:.4f}% of {rep.total_cells})")
            if rep.out_of_0_100_examples:
                add("  [occupancy check] sample locations (values NOT clipped):")
                for ex in rep.out_of_0_100_examples:
                    add(f"      {ex}")
            else:
                add("  [occupancy check] no values outside 0-100 detected")
        add("")

    # ---------------- cross-file consistency ----------------
    add("-" * 78)
    add("CROSS-FILE CONSISTENCY")
    add("-" * 78)

    loaded = [r for r in reports if r.exists and r.timestamps is not None]
    if len(loaded) < 2:
        add("  Insufficient files loaded for comparison.")
    else:
        ref = loaded[0]
        add(f"  Reference file  : {ref.name} ({len(ref.timestamps)} timestamps, "
            f"{len(ref.zone_columns)} zones)")
        for rep in loaded[1:]:
            ts_match = rep.timestamps.equals(ref.timestamps)
            zones_match = rep.zone_columns == ref.zone_columns
            add(f"  {rep.name:<20} timestamps match: {'YES' if ts_match else 'NO':<4}"
                f" zones match: {'YES' if zones_match else 'NO'}")

    # ---------------- expectations ----------------
    add("")
    add("-" * 78)
    add("EXPECTED VALUES (per dataset README)")
    add("-" * 78)
    add(f"  Expected rows        : {EXPECTED_ROWS}")
    add(f"  Expected zones       : {EXPECTED_ZONES}")
    add(f"  Expected start       : {EXPECTED_START}")
    add(f"  Expected end         : {EXPECTED_END}")
    for rep in loaded:
        rows_ok = rep.n_rows == EXPECTED_ROWS
        zones_ok = rep.n_cols == EXPECTED_ZONES
        start_ok = rep.time_min == EXPECTED_START
        end_ok = rep.time_max == EXPECTED_END
        add(f"  {rep.name:<20} rows:{'OK' if rows_ok else 'MISMATCH'} "
            f"zones:{'OK' if zones_ok else 'MISMATCH'} "
            f"start:{'OK' if start_ok else 'MISMATCH'} "
            f"end:{'OK' if end_ok else 'MISMATCH'}")

    # ---------------- summary ----------------
    add("")
    add("=" * 78)
    add("SUMMARY")
    add("=" * 78)
    failed = [r.name for r in reports if not r.exists]
    if failed:
        add(f"  Files that FAILED to load: {', '.join(failed)}")
    else:
        add("  All target files loaded successfully.")
    add("  No files were modified; validation is read-only.")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Validate the UrbanEV hourly dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        print(f"ERROR: data directory not found: {args.data_dir}", file=sys.stderr)
        return 2

    report = build_report(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(report)
    print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
