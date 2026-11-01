"""Tests for src.models.validation_study.

The split logic is the leakage-critical part, so it is tested exhaustively on
synthetic frames. The estimator path is exercised with a tiny sample and the
cheapest candidate only — nothing touches the real dataset or writes files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train_models import PREDICTOR_COLUMNS
from src.models.validation_study import (
    CANDIDATES,
    build_candidate,
    evaluate_on_validation,
    predictors_for,
    temporal_train_valid_split,
)


def _frame(periods: int = 100, zones=("102", "104")) -> pd.DataFrame:
    ts = pd.date_range("2022-09-01", periods=periods, freq="h")
    rows = []
    for z in zones:
        for i, t in enumerate(ts):
            rows.append(
                {
                    "time": t,
                    "zone_id": z,
                    "demand_kwh": 10.0 + (i % 7),
                    "hour": t.hour,
                    "day_of_week": t.dayofweek,
                    "month": t.month,
                    "lag_1": 10.0 + ((i - 1) % 7),
                    "lag_24": 9.0 + (i % 5),
                }
            )
    return pd.DataFrame(rows)


def test_validation_is_the_most_recent_rows_per_zone():
    df = _frame(100)
    search, valid, info = temporal_train_valid_split(df, 0.2)
    for zone in df["zone_id"].unique():
        tr = search[search["zone_id"] == zone]
        va = valid[valid["zone_id"] == zone]
        assert len(va) == 20
        assert len(tr) == 80
        assert tr["time"].max() < va["time"].min()
    assert info["chronological_per_zone"] is True


def test_split_covers_every_row_exactly_once():
    df = _frame(50)
    search, valid, _ = temporal_train_valid_split(df, 0.25)
    assert len(search) + len(valid) == len(df)
    key = lambda d: set(zip(d["zone_id"], d["time"]))
    assert key(search).isdisjoint(key(valid))


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        temporal_train_valid_split(_frame(10), 1.0)
    with pytest.raises(ValueError):
        temporal_train_valid_split(_frame(10), 0.0)


def test_split_handles_tiny_groups_gracefully():
    search, valid, _ = temporal_train_valid_split(_frame(2), 0.5)
    assert len(valid) >= 1 and len(search) >= 1


def test_candidate_zone_mode_uses_more_predictors():
    base = next(c for c in CANDIDATES if c["predictors"] == "base")
    zone = next(c for c in CANDIDATES if c["predictors"] == "zone")
    assert predictors_for(base) == PREDICTOR_COLUMNS
    assert len(predictors_for(zone)) > len(PREDICTOR_COLUMNS)


def test_build_candidate_is_seeded():
    for spec in CANDIDATES:
        model = build_candidate(spec)
        assert model.random_state == 42


def test_hgb_candidate_is_available():
    families = {c["family"] for c in CANDIDATES}
    assert "hgb" in families


def test_evaluate_on_validation_returns_finite_metrics():
    df = _frame(60, zones=("102",))
    zone_cols = {
        c: 1.0
        for c in (
            "zone_charge_count",
            "zone_area_km2",
            "zone_perimeter_km",
            "zone_piles_per_km2",
            "zone_n_stations",
            "zone_longitude",
            "zone_latitude",
        )
    }
    df = df.assign(**zone_cols)
    search, valid, _ = temporal_train_valid_split(df, 0.2)
    spec = {
        "name": "rf_tiny",
        "label": "tiny",
        "family": "rf",
        "params": {"n_estimators": 3, "max_depth": 3, "min_samples_leaf": 2},
        "predictors": "base",
    }
    res = evaluate_on_validation(spec, search, valid)
    assert res["valid"]["rows"] == len(valid)
    assert all(np.isfinite(res["valid"][k]) for k in ("mae", "rmse", "r2"))
