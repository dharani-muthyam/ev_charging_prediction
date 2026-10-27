"""Tests for src.data.validate_dataset.

All test data here is synthesised in memory with pandas. Nothing is written to
the real dataset, and no file under data/ is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate_dataset import (
    check_hourly_continuity,
    validate_dataframe,
)


def _synthetic_frame(periods: int = 5) -> pd.DataFrame:
    """Build a small wide charging frame with a 2-zone hourly index."""
    idx = pd.date_range("2022-09-01 00:00:00", periods=periods, freq="h", name="time")
    return pd.DataFrame(
        {"102": [1.0, 2.0, 3.0, 4.0, 5.0][:periods],
         "104": [0.0, 1.5, 2.5, 3.5, 4.5][:periods]},
        index=idx,
    )


def test_shape_and_index_reported():
    df = _synthetic_frame(5)
    rep = validate_dataframe(df, "synthetic.csv")
    assert rep.n_rows == 5
    assert rep.n_cols == 2
    assert rep.time_min == pd.Timestamp("2022-09-01 00:00:00")
    assert rep.time_max == pd.Timestamp("2022-09-01 04:00:00")


def test_duplicate_timestamps_detected():
    df = _synthetic_frame(3)
    df = pd.concat([df, df.iloc[[1]]])  # duplicate one timestamp
    rep = validate_dataframe(df, "synthetic.csv")
    assert rep.duplicate_timestamps == 1


def test_duplicate_zone_columns_detected():
    df = _synthetic_frame(3)
    df.columns = ["102", "102"]
    rep = validate_dataframe(df, "synthetic.csv")
    assert rep.duplicate_zone_columns == ["102"]


def test_missing_values_counted():
    df = _synthetic_frame(3)
    df.iloc[0, 0] = np.nan
    df.iloc[2, 1] = np.nan
    rep = validate_dataframe(df, "synthetic.csv")
    assert rep.missing_cells == 2


def test_negative_values_counted():
    df = _synthetic_frame(3)
    df.iloc[0, 0] = -5.0
    rep = validate_dataframe(df, "synthetic.csv")
    assert rep.negative_cells == 1


def test_statistics_match_numpy():
    df = _synthetic_frame(5)
    rep = validate_dataframe(df, "synthetic.csv")
    assert rep.cell_min == 0.0
    assert rep.cell_max == 5.0
    assert rep.cell_mean == np.mean([1, 2, 3, 4, 5, 0, 1.5, 2.5, 3.5, 4.5])


def test_occupancy_out_of_range_flagged_not_clipped():
    idx = pd.date_range("2022-09-01", periods=3, freq="h", name="time")
    df = pd.DataFrame({"102": [10.0, 215.0, 50.0], "104": [1.0, 2.0, 3.0]}, index=idx)
    rep = validate_dataframe(df, "occupancy.csv")
    assert rep.out_of_0_100_cells == 1
    assert len(rep.out_of_0_100_examples) == 1
    assert "215.0" in rep.out_of_0_100_examples[0]
    # The original frame must be untouched (no clipping).
    assert df.iloc[1, 0] == 215.0


def test_non_numeric_values_counted():
    idx = pd.date_range("2022-09-01", periods=3, freq="h", name="time")
    df = pd.DataFrame({"102": ["1.0", "bad", "3.0"]}, index=idx)
    rep = validate_dataframe(df, "synthetic.csv")
    assert rep.nonnumeric_cells == 1


def test_hourly_continuity_ok():
    idx = pd.date_range("2022-09-01", periods=10, freq="h")
    ok, _ = check_hourly_continuity(idx)
    assert ok is True


def test_hourly_continuity_gap_detected():
    idx = pd.DatetimeIndex(["2022-09-01 00:00", "2022-09-01 01:00", "2022-09-01 03:00"])
    ok, msg = check_hourly_continuity(idx)
    assert ok is False
    assert "irregular" in msg
