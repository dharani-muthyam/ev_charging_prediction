"""Tests for the evaluation figures (``src.visualization.plots``).

The figures are generated from saved predictions, so these tests build a small
synthetic prediction file and a matching demand table, then check that each
figure is produced with the expected filename and that the underlying
aggregations (error, profile) match numbers computed by hand in the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.visualization import plots


def make_predictions(n_zones: int = 6, n_hours: int = 48) -> pd.DataFrame:
    """Two days of hourly rows for a handful of zones, with real arithmetic."""
    times = pd.date_range("2023-02-01 00:00", periods=n_hours, freq="h")
    zones = [str(1000 + z) for z in range(n_zones)]
    frame = pd.DataFrame(
        [(t, z) for t in times for z in zones], columns=["time", "zone_id"]
    )
    rng = np.random.default_rng(0)
    actual = rng.uniform(0, 200, size=len(frame))
    frame["demand_kwh"] = actual
    # Deliberate offsets so the models are clearly different, not identical.
    frame["baseline_predicted_kwh"] = actual * 0.9 + 5.0
    frame["linear_regression_predicted_kwh"] = actual * 0.95 + 2.0
    frame["random_forest_predicted_kwh"] = actual * 0.98 + 0.5
    return frame


@pytest.fixture()
def predictions() -> pd.DataFrame:
    return make_predictions()


@pytest.fixture()
def demand(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[["time", "zone_id", "demand_kwh"]].copy()


def test_available_models_uses_only_present_columns(predictions):
    models = plots.available_models(predictions)
    assert [label for _, label in models] == [
        "Persistence (lag_1)",
        "Linear Regression",
        "Random Forest",
    ]
    assert plots.available_models(predictions[["demand_kwh"]]) == []


def test_zone_errors_matches_manual_mae(predictions):
    z = plots.zone_errors(predictions, "random_forest_predicted_kwh")
    # The helper returns zones sorted by MAE (worst first); compare by zone.
    assert z["mae"].is_monotonic_decreasing
    got = z.set_index("zone_id")["mae"].sort_index()
    expected = (
        (predictions["demand_kwh"] - predictions["random_forest_predicted_kwh"]).abs()
        .groupby(predictions["zone_id"]).mean()
        .sort_index()
    )
    pd.testing.assert_series_equal(got, expected.rename("mae"), check_names=False)


def test_hourly_profile_averages_each_hour(demand):
    per_hour, _ = plots.hourly_profile(demand)
    assert list(per_hour.index) == list(range(24))
    expected_0 = demand.loc[demand["time"].dt.hour == 0, "demand_kwh"].mean()
    assert per_hour.loc[0] == pytest.approx(expected_0)


def test_all_figures_are_written(predictions, demand, tmp_path):
    predictions.to_csv(tmp_path / "pred.csv", index=False)
    comparison = pd.DataFrame(
        {
            "model": ["Persistence (lag_1)", "Linear Regression", "Random Forest"],
            "mae": [46.98, 43.30, 33.33],
            "rmse": [265.93, 174.06, 151.78],
            "r2": [0.8893, 0.9526, 0.9639],
        }
    )
    comparison.to_csv(tmp_path / "comparison.csv", index=False)
    demand.to_csv(tmp_path / "demand_long.csv", index=False)

    out = tmp_path / "figures"
    facts = plots.generate_all(
        predictions_path=tmp_path / "pred.csv",
        comparison_path=tmp_path / "comparison.csv",
        baseline_metrics_path=tmp_path / "no_baseline.json",  # baseline row comes from the table
        out_dir=out,
        demand_path=tmp_path / "demand_long.csv",
    )

    assert facts["test_rows"] == len(predictions)
    for key, path in facts["figures"].items():
        assert (out / f"fig_{key}.png").is_file(), key
        assert path.endswith(f"fig_{key}.png")

    assert (out / "fig_hourly_demand_profile.png").is_file()
    assert (out / "fig_daily_demand_profile.png").is_file()
    assert 0 <= facts["demand_peak_hour"] <= 23
    assert facts["demand_busiest_day"] in plots.DAY_NAMES


def test_demand_profiles_are_skipped_without_the_table(predictions, tmp_path):
    predictions.to_csv(tmp_path / "pred.csv", index=False)
    pd.DataFrame(
        {
            "model": ["Random Forest"],
            "mae": [1.0],
            "rmse": [2.0],
            "r2": [0.9],
        }
    ).to_csv(tmp_path / "comparison.csv", index=False)

    facts = plots.generate_all(
        predictions_path=tmp_path / "pred.csv",
        comparison_path=tmp_path / "comparison.csv",
        baseline_metrics_path=tmp_path / "none.json",
        out_dir=tmp_path / "figures",
        demand_path=tmp_path / "absent.csv",
    )
    assert "hourly_demand_profile" not in facts["figures"]
    assert "skipped" in facts["demand_profile_figures"]


def test_missing_demand_table_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plots.load_demand_long(tmp_path / "nope.csv")


def test_missing_predictions_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        plots.load_predictions(tmp_path / "nope.csv")
