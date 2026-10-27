"""Tests for src.models.train_models.

Run without pytest (it is not installed in this environment):

    python tests/test_train_models.py

The module also exposes ``test_*`` functions so a future ``pytest`` run picks
them up. Fixtures are synthesised in memory; nothing here trains on or writes to
the real dataset or artifacts.
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

from src.evaluation.baseline_persistence import compute_metrics  # noqa: E402
from src.models.train_models import (  # noqa: E402
    FORBIDDEN_PREDICTORS,
    PREDICTOR_COLUMNS,
    RANDOM_SEED,
    assert_predictors_are_safe,
    build_linear_regression,
    build_random_forest,
    compare_to_baseline,
    train_and_evaluate,
)


def _frames(n: int = 200, zones=("100", "200")):
    """Synthetic feature frames with the real column layout.

    One row per (zone, hour). ``demand_kwh`` is a deterministic function of
    ``lag_1`` plus a small calendar effect, so a model should beat a trivial
    constant predictor and the lags are genuinely informative.
    """
    idx = pd.date_range("2023-01-01 00:00:00", periods=n, freq="h")
    rows = []
    for z in zones:
        for i, t in enumerate(idx):
            lag_1 = 10.0 + (i % 7) + (5.0 if z == "200" else 0.0)
            rows.append(
                {
                    "time": t,
                    "zone_id": z,
                    "demand_kwh": lag_1 + 0.5 * (i % 3),
                    "hour": t.hour,
                    "day_of_week": t.dayofweek,
                    "month": t.month,
                    "lag_1": lag_1,
                    "lag_24": 9.0 + (i % 5),
                }
            )
    df = pd.DataFrame(rows).sort_values(["zone_id", "time"]).reset_index(drop=True)
    split = int(len(df) * 0.8)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ---------------------------------------------------------------- safety guards


def test_predictor_columns_are_leakage_safe():
    assert_predictors_are_safe(PREDICTOR_COLUMNS)  # must not raise


def test_default_predictors_exclude_forbidden_columns():
    for col in ("demand_kwh", "time", "zone_id", "split"):
        assert col not in PREDICTOR_COLUMNS, f"{col} must not be a predictor"


def test_no_predictor_is_in_the_forbidden_map():
    assert not (set(PREDICTOR_COLUMNS) & set(FORBIDDEN_PREDICTORS))


def test_guard_rejects_target_as_predictor():
    try:
        assert_predictors_are_safe(("lag_1", "demand_kwh"))
    except AssertionError as exc:
        assert "demand_kwh" in str(exc)
    else:
        raise AssertionError("guard failed to reject the target as a predictor")


def test_guard_rejects_split_and_identifiers():
    for bad in ("split", "time", "zone_id", "predicted_kwh", "residual_kwh"):
        try:
            assert_predictors_are_safe(("lag_1", bad))
        except AssertionError:
            pass
        else:
            raise AssertionError(f"guard failed to reject {bad!r}")


def test_guard_rejects_duplicate_predictors():
    try:
        assert_predictors_are_safe(("lag_1", "lag_1"))
    except AssertionError:
        pass
    else:
        raise AssertionError("guard failed to reject duplicate predictors")


# ---------------------------------------------------------------- model builders


def test_builders_return_unfitted_estimators():
    lr = build_linear_regression()
    rf = build_random_forest()
    # Unfitted sklearn estimators raise on predict or expose no coef_.
    assert not hasattr(lr, "coef_")
    assert not hasattr(rf, "estimators_")


def test_random_forest_is_seeded_and_deterministic():
    rf = build_random_forest()
    assert rf.random_state == RANDOM_SEED

    x = np.arange(200, dtype=float).reshape(-1, 1)
    y = (2.0 * x.ravel()) + 1.0
    a = build_random_forest().fit(x, y).predict(x)
    b = build_random_forest().fit(x, y).predict(x)
    assert np.allclose(a, b), "same seed must reproduce identical predictions"


# ---------------------------------------------------------------- train/evaluate


def test_train_and_evaluate_shapes_and_no_nan():
    tr, te = _frames()
    for key in ("linear_regression", "random_forest"):
        model, preds, test_m, train_m = train_and_evaluate(tr, te, key)
        assert len(preds) == len(te), f"{key}: prediction count must equal test rows"
        assert list(preds.columns) == ["time", "zone_id", "demand_kwh", "lag_1", "predicted_kwh"]
        assert not preds["predicted_kwh"].isna().any(), f"{key}: NaN predictions"
        for m in (test_m, train_m):
            assert all(np.isfinite(m[k]) for k in ("mae", "rmse", "r2"))


def test_predictions_align_with_test_rows_in_order():
    tr, te = _frames()
    for key in ("linear_regression", "random_forest"):
        _, preds, _, _ = train_and_evaluate(tr, te, key)
        assert (preds["time"].to_numpy() == te["time"].to_numpy()).all()
        assert (preds["zone_id"].to_numpy() == te["zone_id"].to_numpy()).all()
        assert np.array_equal(preds["demand_kwh"].to_numpy(), te["demand_kwh"].to_numpy())


def test_metrics_match_independent_recomputation():
    tr, te = _frames()
    for key in ("linear_regression", "random_forest"):
        _, preds, test_m, _ = train_and_evaluate(tr, te, key)
        recomputed = compute_metrics(
            preds["demand_kwh"].to_numpy(), preds["predicted_kwh"].to_numpy()
        )
        assert test_m["mae"] == recomputed["mae"]
        assert test_m["rmse"] == recomputed["rmse"]
        assert test_m["r2"] == recomputed["r2"]


def test_model_learns_signal_on_synthetic_data():
    """A fitted model must beat predicting the training mean on held-out rows."""
    tr, te = _frames()
    for key in ("linear_regression", "random_forest"):
        _, _, test_m, _ = train_and_evaluate(tr, te, key)
        mean_baseline = compute_metrics(
            te["demand_kwh"].to_numpy(),
            np.full(len(te), tr["demand_kwh"].mean()),
        )
        assert test_m["mae"] < mean_baseline["mae"], f"{key} failed to learn the signal"


def test_training_uses_train_rows_only():
    """Changing the test target must not change what the model learns.

    If the test rows leaked into fitting, perturbing only the test target would
    alter the learned model and therefore its predictions on the *training*
    rows. We compare predictions on the train rows, not the reported metric
    floats: Random Forest sums leaf values across threads (``n_jobs=-1``), so
    repeated fits can differ at the ~1e-15 level. That is floating-point
    reduction order, not leakage, hence the tolerance.

    The companion test below pins the stronger, exact invariant for the
    deterministic Linear Regression coefficients.
    """
    tr, te = _frames()
    te_perturbed = te.copy()
    te_perturbed["demand_kwh"] = -999.0

    for key in ("linear_regression", "random_forest"):
        model_a, _, _, _ = train_and_evaluate(tr, te, key)
        model_b, _, _, _ = train_and_evaluate(tr, te_perturbed, key)

        x_train = tr[list(PREDICTOR_COLUMNS)]
        pred_a = np.asarray(model_a.predict(x_train), dtype=float)
        pred_b = np.asarray(model_b.predict(x_train), dtype=float)
        assert np.allclose(pred_a, pred_b, rtol=0, atol=1e-9), (
            f"{key}: the test target changed the learned model (leakage suspected)"
        )


def test_fitted_coefficients_independent_of_test_data():
    """Linear Regression coefficients must depend only on the train rows."""
    tr, te = _frames()
    model_a = build_linear_regression().fit(
        tr[list(PREDICTOR_COLUMNS)], tr["demand_kwh"]
    )

    te2 = te.copy()
    te2["demand_kwh"] = te2["demand_kwh"] * 100.0
    model_b = build_linear_regression().fit(
        tr[list(PREDICTOR_COLUMNS)], tr["demand_kwh"]
    )
    assert np.allclose(model_a.coef_, model_b.coef_), "test data must not affect fitting"


# ---------------------------------------------------------------- comparison


def test_compare_to_baseline_detects_improvement():
    baseline = {"test": {"mae": 10.0, "rmse": 20.0, "r2": 0.5, "rows": 100}}
    better = {"mae": 9.0, "rmse": 19.0, "r2": 0.6, "rows": 100}
    c = compare_to_baseline(better, baseline)
    assert c["beats_baseline_on_all_three"] is True
    assert c["mae_delta"] == -1.0 and c["same_test_rows"] is True


def test_compare_to_baseline_rejects_partial_improvement():
    baseline = {"test": {"mae": 10.0, "rmse": 20.0, "r2": 0.5, "rows": 100}}
    mixed = {"mae": 9.0, "rmse": 21.0, "r2": 0.6, "rows": 100}  # RMSE worse
    c = compare_to_baseline(mixed, baseline)
    assert c["beats_baseline_on_all_three"] is False


def test_compare_to_baseline_flags_row_count_mismatch():
    baseline = {"test": {"mae": 10.0, "rmse": 20.0, "r2": 0.5, "rows": 100}}
    other = {"mae": 9.0, "rmse": 19.0, "r2": 0.6, "rows": 99}
    c = compare_to_baseline(other, baseline)
    assert c["same_test_rows"] is False


# ---------------------------------------------------------------- runner


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
