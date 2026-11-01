"""Tests for src.features.zone_features.

Synthetic metadata only — no file under data/ is read or written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.zone_features import (
    FEATURE_COLUMNS,
    attach_zone_features,
    load_zone_table,
)


def _metadata(tmp_path):
    zones = pd.DataFrame(
        {
            "TAZID": [102, 104],
            "longitude": [113.9, 114.0],
            "latitude": [22.5, 22.6],
            "charge_count": [40, 10],
            "area": [1_000_000.0, 250_000.0],
            "perimeter": [4_000.0, 2_000.0],
        }
    )
    stations = pd.DataFrame(
        {
            "station_id": [1001, 1002, 1003, 1004],
            "TAZID": [102, 102, 102, 104],
            "longitude": [113.9] * 4,
            "latitude": [22.5] * 4,
            "charge_count": [20, 20, 5, 10],
        }
    )
    zp = tmp_path / "zone.csv"
    sp = tmp_path / "station.csv"
    zones.to_csv(zp, index=False)
    stations.to_csv(sp, index=False)
    return zp, sp


def _features(splits_names=("102", "104"), n=3):
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    rows = [
        {"time": t, "zone_id": z, "demand_kwh": 1.0, "lag_1": 1.0}
        for z in splits_names
        for t in ts
    ]
    return pd.DataFrame(rows)


def test_zone_table_is_one_row_per_zone(tmp_path):
    table = load_zone_table(*_metadata(tmp_path))
    assert len(table) == 2
    assert not table["zone_id"].duplicated().any()
    assert set(FEATURE_COLUMNS).issubset(table.columns)


def test_zone_table_converts_units_and_derives_density(tmp_path):
    table = load_zone_table(*_metadata(tmp_path)).set_index("zone_id")
    assert table.loc["102", "zone_area_km2"] == pytest.approx(1.0)
    assert table.loc["104", "zone_area_km2"] == pytest.approx(0.25)
    assert table.loc["102", "zone_perimeter_km"] == pytest.approx(4.0)
    # 40 piles over 1 km^2
    assert table.loc["102", "zone_piles_per_km2"] == pytest.approx(40.0)
    # station count is a unique-station count, not a pile sum
    assert table.loc["102", "zone_n_stations"] == 3
    assert table.loc["104", "zone_n_stations"] == 1


def test_attach_preserves_row_count_and_order(tmp_path):
    feat = _features()
    out, report = attach_zone_features(feat, load_zone_table(*_metadata(tmp_path)))
    assert len(out) == len(feat)
    assert report["rows_before"] == report["rows_after"] == len(feat)
    assert out["zone_id"].tolist() == feat["zone_id"].tolist()
    assert out["time"].tolist() == feat["time"].tolist()
    assert not out[list(FEATURE_COLUMNS)].isna().any().any()


def test_attach_is_many_to_one_no_fan_out(tmp_path):
    """Three stations in zone 102 must not triple that zone's rows."""
    feat = _features()
    out, _ = attach_zone_features(feat, load_zone_table(*_metadata(tmp_path)))
    assert out.groupby("zone_id").size().tolist() == [3, 3]


def test_attach_rejects_unmatched_zone(tmp_path):
    feat = _features(splits_names=("102", "999"))
    with pytest.raises(AssertionError, match="no zone metadata"):
        attach_zone_features(feat, load_zone_table(*_metadata(tmp_path)))


def test_zone_features_are_static_per_zone(tmp_path):
    """Within a zone every timestamp must see the same attribute values."""
    out, _ = attach_zone_features(_features(), load_zone_table(*_metadata(tmp_path)))
    for col in FEATURE_COLUMNS:
        assert out.groupby("zone_id")[col].nunique().max() == 1


def test_zone_features_add_no_identifiers(tmp_path):
    """The join must add attributes only — no TAZID/station_id columns."""
    feat = _features()
    out, _ = attach_zone_features(feat, load_zone_table(*_metadata(tmp_path)))
    assert set(out.columns) - set(feat.columns) == set(FEATURE_COLUMNS)


def test_missing_metadata_file_raises(tmp_path):
    zp, _ = _metadata(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_zone_table(zp, tmp_path / "does-not-exist.csv")


def test_zone_ids_are_strings_after_join(tmp_path):
    out, _ = attach_zone_features(_features(), load_zone_table(*_metadata(tmp_path)))
    assert all(isinstance(v, str) for v in out["zone_id"])
    assert np.all(out["zone_id"].to_numpy() == np.array(["102", "102", "102", "104", "104", "104"]))
