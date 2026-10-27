"""Tests for src.features.demand_lags.

Run without pytest (it is not installed in this environment):

    python tests/test_demand_lags.py

The module also exposes ``test_*`` functions so a future ``pytest`` run picks
them up automatically. Fixtures are synthesised in memory; nothing here writes
to ``data/``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.demand_lags import (  # noqa: E402
    LAG_HOURS,
    TARGET_COLUMN,
    NON_PREDICTOR_COLUMNS,
    add_lag_features,
    lag_column_names,
)


# --------------------------------------------------------------------------
# Fixture: a synthetic long frame, two zones, contiguous hourly timestamps.
# --------------------------------------------------------------------------


def _long(periods: int = 48, zones=("102", "104")) -> pd.DataFrame:
    """Zone-major long frame. Zone '102' values are 0..n-1, zone '104' +1000.

    Deliberately *not* sorted: zone '104' is placed first so that a buggy global
    shift would be caught (it would fill zone '102' from zone '104').
    """
    idx = pd.date_range("2022-09-01 00:00:00", periods=periods, freq="h")
    frames = []
    for i, z in enumerate(sorted(zones, reverse=True)):
        frames.append(
            pd.DataFrame(
                {
                    "time": idx,
                    "zone_id": z,
                    "demand_kwh": np.arange(periods, dtype=float) + 1000.0 * i,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Lag alignment
# --------------------------------------------------------------------------


def test_lag_columns_are_added():
    out = add_lag_features(_long(periods=48))
    for name in lag_column_names():
        assert name in out.columns, f"{name} missing"
    assert list(lag_column_names()) == ["lag_1", "lag_24"]


def test_lag_1_is_previous_hour():
    """lag_1 at time t must equal demand at t-1, within the same zone."""
    out = add_lag_features(_long(periods=48))
    for zone in out["zone_id"].unique():
        g = out[out["zone_id"] == zone].reset_index(drop=True)
        # Row 0 has no history; every later row's lag_1 is the previous row's demand.
        assert np.isnan(g.loc[0, "lag_1"])
        for t in range(1, len(g)):
            assert g.loc[t, "lag_1"] == g.loc[t - 1, "demand_kwh"], t


def test_lag_24_is_previous_day_same_hour():
    """lag_24 at time t must equal demand at t-24, within the same zone."""
    out = add_lag_features(_long(periods=48))
    for zone in out["zone_id"].unique():
        g = out[out["zone_id"] == zone].reset_index(drop=True)
        for t in range(len(g)):
            if t < 24:
                assert np.isnan(g.loc[t, "lag_24"])
            else:
                assert g.loc[t, "lag_24"] == g.loc[t - 24, "demand_kwh"], t


def test_lag_values_match_manual_shift():
    """Whole-frame check against an independent pandas shift."""
    df = _long(periods=48)
    out = add_lag_features(df)
    expected = (
        df.sort_values(["zone_id", "time"])
        .groupby("zone_id")["demand_kwh"]
        .shift(1)
        .reset_index(drop=True)
    )
    got = out["lag_1"].reset_index(drop=True)
    assert np.allclose(got, expected, equal_nan=True)


def test_custom_lag_offsets_are_honoured():
    out = add_lag_features(_long(periods=48), lag_hours=(2, 3))
    assert "lag_2" in out.columns and "lag_3" in out.columns
    for zone in out["zone_id"].unique():
        g = out[out["zone_id"] == zone].reset_index(drop=True)
        for t in range(len(g)):
            if t >= 2:
                assert g.loc[t, "lag_2"] == g.loc[t - 2, "demand_kwh"], t
            if t >= 3:
                assert g.loc[t, "lag_3"] == g.loc[t - 3, "demand_kwh"], t


def test_columns_are_ordered_zone_then_time():
    out = add_lag_features(_long(periods=10))
    z = out["zone_id"].tolist()
    assert z == sorted(z)
    for _, group in out.groupby("zone_id", sort=False):
        assert group["time"].is_monotonic_increasing


def test_original_frame_is_not_mutated():
    df = _long(periods=10)
    before = df.copy()
    add_lag_features(df)
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------------
# Zone isolation
# --------------------------------------------------------------------------


def test_zone_isolation_lag_1_never_crosses_zones():
    """No zone's first row may inherit another zone's value."""
    out = add_lag_features(_long(periods=48))
    for zone in out["zone_id"].unique():
        first = out[out["zone_id"] == zone].iloc[0]
        assert np.isnan(first["lag_1"]), f"zone {zone} lag_1 leaked in"
        assert np.isnan(first["lag_24"]), f"zone {zone} lag_24 leaked in"


def test_zone_isolation_lag_values_come_from_own_zone_only():
    """Every non-NaN lag must equal a demand value belonging to the same zone."""
    out = add_lag_features(_long(periods=48))
    for zone in out["zone_id"].unique():
        g = out[out["zone_id"] == zone]
        zone_values = set(g["demand_kwh"].tolist())
        for col in lag_column_names():
            observed = set(g[col].dropna().tolist())
            assert observed.issubset(zone_values), f"{col} bled across zones"


def test_zone_isolation_differs_from_global_shift():
    """A per-zone shift must NOT equal a naive global shift at zone boundaries."""
    out = add_lag_features(_long(periods=48))
    global_shift = _long(periods=48)["demand_kwh"].shift(1).reset_index(drop=True)
    per_zone = out["lag_1"].reset_index(drop=True)
    # The very first row of the second zone distinguishes the two.
    second_zone = out["zone_id"].unique()[1]
    first_of_second = out.index[out["zone_id"] == second_zone][0]
    assert np.isnan(per_zone.iloc[first_of_second])
    assert not np.isnan(global_shift.iloc[first_of_second])


def test_warmup_rows_per_zone_equal_max_lag():
    """Each zone loses exactly `max(lag_hours)` rows at the head."""
    df = _long(periods=48)
    out = add_lag_features(df)
    n = len(df) // df["zone_id"].nunique()
    warmup = out[lag_column_names()].isna().any(axis=1).sum()
    assert warmup == df["zone_id"].nunique() * max(LAG_HOURS)


# --------------------------------------------------------------------------
# Leakage prevention
# --------------------------------------------------------------------------


def test_no_future_information_in_predictors():
    """lag_1/lag_24 at t may only reference rows strictly before t."""
    out = add_lag_features(_long(periods=48))
    for zone in out["zone_id"].unique():
        g = out[out["zone_id"] == zone].reset_index(drop=True)
        for t in range(len(g)):
            if t >= 1:
                assert g.loc[t, "lag_1"] == g.loc[t - 1, "demand_kwh"]
            if t >= 24:
                assert g.loc[t, "lag_24"] == g.loc[t - 24, "demand_kwh"]
            # No lag may ever equal the *current* target unless it is a
            # genuine coincidence of the series (guarded by the shifts above).


def test_target_is_not_a_contemporaneous_feature():
    """demand_kwh is the label; it must be excluded from the predictor set."""
    out = add_lag_features(_long(periods=48))
    predictors = [c for c in out.columns if c not in NON_PREDICTOR_COLUMNS]
    assert TARGET_COLUMN not in predictors
    assert "lag_1" in predictors and "lag_24" in predictors


def test_lags_are_independent_of_row_order():
    """Shuffling the input rows must not change the lag values."""
    df = _long(periods=48)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = add_lag_features(df).sort_values(["zone_id", "time"]).reset_index(drop=True)
    b = add_lag_features(shuffled).sort_values(["zone_id", "time"]).reset_index(drop=True)
    for col in lag_column_names():
        assert np.allclose(a[col], b[col], equal_nan=True), col


def test_split_cannot_change_lag_values():
    """Lags are a pure function of history; a later split must reproduce them.

    Truncating the history at an earlier point and recomputing must give the
    same lag for the surviving rows — proof the values do not depend on rows
    that are "in the future" relative to the split.
    """
    df = _long(periods=48)
    full = add_lag_features(df)
    cutoff = 40
    early = df[df["time"] < df["time"].unique()[cutoff]]
    early_lagged = add_lag_features(early)

    key = ["zone_id", "time"]
    merged = full[full["time"] < df["time"].unique()[cutoff]].merge(
        early_lagged, on=key, suffixes=("_full", "_early")
    )
    for col in lag_column_names():
        assert np.allclose(
            merged[f"{col}_full"], merged[f"{col}_early"], equal_nan=True
        ), f"{col} depends on future rows"


def test_dropping_warmup_leaves_no_nan():
    """The documented policy: drop, do not impute."""
    out = add_lag_features(_long(periods=48))
    kept = out.dropna(subset=lag_column_names())
    assert len(kept) == len(out) - out["zone_id"].nunique() * max(LAG_HOURS)
    assert not kept[lag_column_names()].isna().any().any()


# --------------------------------------------------------------------------
# Minimal runner so this file works without pytest
# --------------------------------------------------------------------------


def _run_all() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures: list[str] = []
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - test harness
            failures.append(f"{name}\n{traceback.format_exc()}")
        else:
            print(f"PASS  {name}")

    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
