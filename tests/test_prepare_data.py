"""Tests for src.preprocessing.prepare_data.

All fixtures here are synthesised in memory. Nothing is written to data/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.prepare_data import (
    add_calendar_features,
    chronological_split,
    validate_wide,
    wide_to_long,
)


def _wide(periods: int = 10, zones=("102", "104")) -> pd.DataFrame:
    idx = pd.date_range("2022-09-01 00:00:00", periods=periods, freq="h", name="time")
    data = {z: np.arange(periods, dtype=float) + i for i, z in enumerate(zones)}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------- reshaping


def test_wide_to_long_shape_and_columns():
    wide = _wide(periods=10, zones=("102", "104"))
    long_df = wide_to_long(wide)
    assert list(long_df.columns) == ["time", "zone_id", "demand_kwh"]
    assert len(long_df) == 20  # 10 timestamps x 2 zones


def test_wide_to_long_preserves_values_exactly():
    wide = _wide(periods=5, zones=("102",))
    long_df = wide_to_long(wide)
    original = wide["102"].to_numpy()
    assert np.array_equal(long_df["demand_kwh"].to_numpy(), original)


def test_wide_to_long_sorted_by_zone_then_time():
    wide = _wide(periods=6, zones=("104", "102"))  # deliberately reversed
    long_df = wide_to_long(wide)
    assert long_df["zone_id"].tolist() == sorted(long_df["zone_id"].tolist())
    for _, group in long_df.groupby("zone_id"):
        assert group["time"].is_monotonic_increasing


def test_wide_to_long_is_lossless_roundtrip():
    wide = _wide(periods=8, zones=("102", "104", "105"))
    long_df = wide_to_long(wide)
    back = long_df.pivot(index="time", columns="zone_id", values="demand_kwh")
    back = back[wide.columns]
    pd.testing.assert_frame_equal(back, wide, check_names=False)


# ---------------------------------------------------------------- calendar


def test_calendar_features_correct():
    wide = _wide(periods=1, zones=("102",))
    long_df = add_calendar_features(wide_to_long(wide))
    assert long_df["hour"].iloc[0] == 0
    assert long_df["day_of_week"].iloc[0] == pd.Timestamp("2022-09-01").dayofweek  # Thursday=3
    assert long_df["month"].iloc[0] == 9


def test_calendar_uses_row_own_timestamp_only():
    """A row's features must depend only on its own timestamp."""
    wide = _wide(periods=3, zones=("102",))
    featured = add_calendar_features(wide_to_long(wide))
    for _, row in featured.iterrows():
        assert row["hour"] == pd.Timestamp(row["time"]).hour
        assert row["month"] == pd.Timestamp(row["time"]).month


# ---------------------------------------------------------------- split


def test_chronological_split_is_80_20_per_zone():
    wide = _wide(periods=10, zones=("102", "104"))
    featured = add_calendar_features(wide_to_long(wide))
    train, test, summaries = chronological_split(featured, 0.8)
    for s in summaries:
        assert s.n_train == 8
        assert s.n_test == 2
    assert len(train) == 16
    assert len(test) == 4


def test_chronological_split_is_not_shuffled():
    """Every test timestamp must be later than every train timestamp, per zone."""
    wide = _wide(periods=10, zones=("102", "104"))
    featured = add_calendar_features(wide_to_long(wide))
    train, test, _ = chronological_split(featured, 0.8)
    for zone in ("102", "104"):
        tr = train[train["zone_id"] == zone]["time"]
        te = test[test["zone_id"] == zone]["time"]
        assert tr.max() < te.min()


def test_chronological_split_covers_all_rows_without_overlap():
    wide = _wide(periods=9, zones=("102", "104"))
    featured = add_calendar_features(wide_to_long(wide))
    train, test, _ = chronological_split(featured, 0.8)
    assert len(train) + len(test) == len(featured)
    key = lambda d: set(zip(d["zone_id"], d["time"]))
    assert key(train).isdisjoint(key(test))


def test_chronological_split_no_test_row_before_train_row():
    """Guard against temporal leakage, checked per zone."""
    wide = _wide(periods=20, zones=("102", "104"))
    featured = add_calendar_features(wide_to_long(wide))
    train, test, _ = chronological_split(featured, 0.8)
    for zone in featured["zone_id"].unique():
        tr_max = train[train["zone_id"] == zone]["time"].max()
        te_min = test[test["zone_id"] == zone]["time"].min()
        assert tr_max < te_min


def test_split_rejects_invalid_fraction():
    wide = _wide(periods=10, zones=("102",))
    featured = add_calendar_features(wide_to_long(wide))
    with pytest.raises(ValueError):
        chronological_split(featured, 1.5)
    with pytest.raises(ValueError):
        chronological_split(featured, 0.0)


# ---------------------------------------------------------------- validation


def test_validate_wide_accepts_clean_frame():
    validate_wide(_wide(periods=5, zones=("102",)))  # should not raise


def test_validate_wide_rejects_gap():
    wide = _wide(periods=5, zones=("102",))
    wide = wide.drop(wide.index[2])  # introduce a 2-hour gap
    with pytest.raises(ValueError, match="hourly/contiguous"):
        validate_wide(wide)


def test_validate_wide_rejects_duplicate_timestamps():
    wide = _wide(periods=3, zones=("102",))
    wide = pd.concat([wide, wide.iloc[[1]]])
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_wide(wide)


def test_validate_wide_rejects_duplicate_columns():
    wide = _wide(periods=3, zones=("102", "102"))
    with pytest.raises(ValueError, match="Duplicate zone columns"):
        validate_wide(wide)


# ------------------------------------------------- float / string round-trip


def test_zone_id_roundtrips_as_string(tmp_path):
    """zone_id must stay textual through a CSV write/read cycle."""
    from src.preprocessing.prepare_data import _dump_csv, load_processed

    long_df = wide_to_long(_wide(periods=3, zones=("102", "1000")))
    assert long_df["zone_id"].dtype == object
    path = tmp_path / "out.csv"
    _dump_csv(long_df, path)
    back = load_processed(path)
    assert all(isinstance(v, str) for v in back["zone_id"])
    assert back["zone_id"].tolist() == long_df["zone_id"].tolist()


def test_demand_kwh_float64_exact_roundtrip(tmp_path):
    """CSV writing must preserve float64 values bit-exactly (repr-based)."""
    from src.preprocessing.prepare_data import _dump_csv, load_processed

    idx = pd.date_range("2022-09-01", periods=5, freq="h", name="time")
    nasty = [1.1666666666666665, 0.1, 3.3000000000000003, 56.29166666666666, 1.8189894035458565e-12]
    wide = pd.DataFrame({"102": nasty}, index=idx)
    long_df = wide_to_long(wide)

    path = tmp_path / "out.csv"
    _dump_csv(long_df, path)
    back = load_processed(path)

    assert np.array_equal(back["demand_kwh"].to_numpy(), wide["102"].to_numpy())


def test_processed_rows_sorted_string_zone_then_time(tmp_path):
    """Rows must be grouped by textual zone_id, then ordered by time."""
    from src.preprocessing.prepare_data import _dump_csv, load_processed

    long_df = wide_to_long(_wide(periods=5, zones=("1000", "102", "104")))
    path = tmp_path / "out.csv"
    _dump_csv(long_df, path)
    back = load_processed(path)

    z = back["zone_id"].tolist()
    assert z == sorted(z)  # textual, not numeric, ordering
    for _, group in back.groupby("zone_id", sort=False):
        assert group["time"].is_monotonic_increasing
