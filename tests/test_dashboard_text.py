"""Tests for the dashboard's plain-language helpers (``app.dashboard_text``).

These functions only *describe* numbers that the offline pipeline already
wrote, so the tests check that:
  * the wording matches the numbers passed in (no hard-coded claims),
  * missing or malformed values are handled instead of producing fake output,
  * the real saved artefacts still carry the documented reference metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.dashboard_text import (
    BASELINE_LABEL,
    column_context,
    describe_metric,
    explain_comparison,
    format_kwh,
    improvement_vs_baseline,
    validate_demand_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def make_table() -> pd.DataFrame:
    """The recorded test-set metrics, laid out like the dashboard table."""
    return pd.DataFrame(
        {
            "Model": [BASELINE_LABEL, "Linear Regression", "Random Forest"],
            "MAE (kWh)": [46.983300, 43.299037, 33.327187],
            "RMSE (kWh)": [265.933875, 174.055987, 151.777197],
            "R²": [0.889306, 0.952581, 0.963943],
        }
    )


# --------------------------------------------------------------------------
# Formatting and metric text
# --------------------------------------------------------------------------


def test_format_kwh_always_carries_the_unit():
    assert format_kwh(33.3272) == "33.33 kWh"
    assert format_kwh(1234.5, 1) == "1,234.5 kWh"


@pytest.mark.parametrize("key", ["MAE (kWh)", "RMSE (kWh)", "R²"])
def test_every_metric_has_a_short_explanation(key):
    text = describe_metric(key)
    assert text and text.endswith(".")


def test_unknown_metric_returns_empty_string():
    assert describe_metric("nope") == ""


# --------------------------------------------------------------------------
# Reading one model's row
# --------------------------------------------------------------------------


def test_column_context_returns_the_rows_own_numbers():
    ctx = column_context(make_table(), "Random Forest")
    assert ctx["mae"] == pytest.approx(33.327187)
    assert ctx["r2"] == pytest.approx(0.963943)


def test_column_context_accepts_a_frame_indexed_by_model():
    """Regression: the dashboard passes ``set_index('Model')``, not a column."""
    indexed = make_table().set_index("Model")
    ctx = column_context(indexed, "Linear Regression")
    assert ctx is not None
    assert ctx["mae"] == pytest.approx(43.299037)


def test_column_context_rejects_a_duplicated_index():
    indexed = pd.concat([make_table(), make_table()]).set_index("Model")
    assert column_context(indexed, "Random Forest") is None


@pytest.mark.parametrize(
    "table, name",
    [
        (pd.DataFrame(), "Random Forest"),
        (None, "Random Forest"),
        (make_table(), "Not A Model"),
    ],
)
def test_column_context_is_none_when_unusable(table, name):
    assert column_context(table, name) is None


def test_column_context_ignores_non_numeric_mae():
    table = pd.DataFrame({"Model": ["X"], "MAE (kWh)": ["n/a"], "R²": [0.5]})
    assert column_context(table, "X") is None


# --------------------------------------------------------------------------
# Baseline comparison
# --------------------------------------------------------------------------


def test_improvement_vs_baseline_is_negative_when_error_drops():
    gain = improvement_vs_baseline(make_table(), "Random Forest")
    assert gain["mae_delta"] == pytest.approx(33.327187 - 46.983300)
    assert gain["mae_pct"] == pytest.approx(
        (33.327187 - 46.983300) / 46.983300 * 100.0
    )
    assert gain["mae_pct"] < 0


def test_improvement_vs_baseline_missing_model_returns_none():
    assert improvement_vs_baseline(make_table(), "Missing Model") is None


def test_improvement_vs_baseline_without_baseline_returns_none():
    table = make_table()
    assert improvement_vs_baseline(table[table["Model"] != BASELINE_LABEL], "Random Forest") is None


def test_improvement_vs_baseline_guards_zero_baseline():
    table = make_table()
    table.loc[0, "MAE (kWh)"] = 0.0
    assert improvement_vs_baseline(table, "Random Forest") is None


# --------------------------------------------------------------------------
# Explanation sentences
# --------------------------------------------------------------------------


def test_explanation_names_the_actual_best_model():
    lines = explain_comparison(make_table())
    text = " ".join(lines)
    # Random Forest is genuinely best on all three metrics in this table.
    assert "Random Forest" in text
    assert "33.33 kWh" in text
    assert "0.9639" in text
    assert "all three metrics" in text


def test_explanation_does_not_hardcode_a_winner():
    """Swap which model is best and the sentence must follow the data."""
    table = make_table()
    table.loc[1, ["MAE (kWh)", "RMSE (kWh)", "R²"]] = [10.0, 20.0, 0.99]
    text = " ".join(explain_comparison(table))
    assert "Linear Regression" in text
    assert "33.33 kWh" not in text


def test_explanation_reports_a_split_decision_without_overclaiming():
    table = make_table()
    # Make Linear Regression best on MAE only, Random Forest best on RMSE and R².
    table.loc[1, "MAE (kWh)"] = 10.0
    lines = explain_comparison(table)
    text = " ".join(lines)
    assert "No single model wins on every metric" in text
    assert "best on all three" not in text


def test_explanation_of_empty_table_is_empty():
    assert explain_comparison(pd.DataFrame()) == []


def test_explanation_mentions_percent_vs_baseline():
    text = " ".join(explain_comparison(make_table()))
    assert "lower than repeating the previous hour's demand" in text


# --------------------------------------------------------------------------
# Demand input validation (lag_1 / lag_24 must be >= 0 kWh)
# --------------------------------------------------------------------------


def test_valid_positive_demand_inputs_are_accepted():
    assert validate_demand_inputs({"lag_1": 50.0, "lag_24": 45.0}) == []


def test_zero_demand_is_valid():
    """A zone that delivered nothing last hour is a real, legitimate case."""
    assert validate_demand_inputs({"lag_1": 0, "lag_24": 0}) == []
    assert validate_demand_inputs({"lag_1": 0.0, "lag_24": 123.4}) == []


def test_negative_lag_1_is_rejected():
    problems = validate_demand_inputs({"lag_1": -25.0, "lag_24": 45.0})
    assert len(problems) == 1
    assert "Last hour's demand" in problems[0]
    assert "0 kWh or more" in problems[0]
    assert "-25.00" in problems[0]


def test_negative_lag_24_is_rejected():
    problems = validate_demand_inputs({"lag_1": 50.0, "lag_24": -1})
    assert len(problems) == 1
    assert "Demand at this hour yesterday" in problems[0]


def test_both_negative_inputs_produce_two_messages():
    problems = validate_demand_inputs({"lag_1": -5, "lag_24": -10})
    assert len(problems) == 2


def test_non_numeric_and_non_finite_are_rejected():
    for bad in ("abc", None, float("nan"), float("inf")):
        assert validate_demand_inputs({"lag_1": bad, "lag_24": 10}) != []


def test_keys_not_present_are_ignored():
    """Only the demand inputs are checked; calendar fields are not validated here."""
    assert validate_demand_inputs({"hour": -3}) == []


def test_negative_value_is_not_passed_to_the_model(monkeypatch):
    """The guard runs before prediction: a rejected input never reaches predict."""

    class ExplodingModel:
        def predict(self, _features):  # pragma: no cover - must never run
            raise AssertionError("model.predict must not be called for invalid input")

    feature_values = {"hour": 18, "day_of_week": 4, "month": 9, "lag_1": -25.0, "lag_24": 45.0}
    problems = validate_demand_inputs(feature_values)
    assert problems, "negative lag_1 must be rejected"

    # Mirror the dashboard: only call the model when validation passes.
    if not problems:
        ExplodingModel().predict(feature_values)


# --------------------------------------------------------------------------
# The real saved artefacts
# --------------------------------------------------------------------------


def test_saved_reports_still_hold_the_reference_metrics():
    """Guard the documented numbers; this reads files, it never writes them."""
    baseline = json.loads((ROOT / "reports" / "baseline_metrics.json").read_text("utf-8"))["test"]
    models = json.loads((ROOT / "reports" / "model_metrics.json").read_text("utf-8"))["models"]

    assert baseline["mae"] == pytest.approx(46.9833, abs=1e-4)
    assert baseline["rmse"] == pytest.approx(265.9339, abs=1e-4)
    assert baseline["r2"] == pytest.approx(0.8893, abs=1e-4)

    assert models["linear_regression"]["test"]["mae"] == pytest.approx(43.2990, abs=1e-4)
    assert models["linear_regression"]["test"]["rmse"] == pytest.approx(174.0560, abs=1e-4)
    assert models["linear_regression"]["test"]["r2"] == pytest.approx(0.9526, abs=1e-4)

    assert models["random_forest"]["test"]["mae"] == pytest.approx(33.3272, abs=1e-4)
    assert models["random_forest"]["test"]["rmse"] == pytest.approx(151.7772, abs=1e-4)
    assert models["random_forest"]["test"]["r2"] == pytest.approx(0.9639, abs=1e-4)
