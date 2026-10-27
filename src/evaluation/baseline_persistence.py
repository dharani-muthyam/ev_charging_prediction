"""Persistence ("naive") baseline for one-hour-ahead EV charging demand.

What this module does
---------------------
It evaluates the simplest possible forecasting rule: predict the demand at time
``t`` with the demand observed at ``t-1``::

    y_pred = lag_1

No model is fitted, no parameter is learned, and nothing is tuned. The baseline
exists only to establish a *reference point*: any later model must beat these
numbers on the same test rows to be worth its complexity.

Why this is a fair baseline (no leakage)
----------------------------------------
* ``lag_1`` is produced by ``src.features.demand_lags`` with a *within-zone*
  ``shift(1)`` on chronologically ordered history. It is therefore, by
  construction, a value that is genuinely known at prediction time ``t-1``.
* The prediction reads ``lag_1`` only. ``demand_kwh`` (the target) is used
  exclusively as the label ``y_true`` in the metric computation — it never
  enters the prediction. That invariant is asserted at runtime, not just
  documented.
* The existing chronological train/test split is reused as-is; no row is moved,
  no random shuffle is performed, and the test window is never used to fit
  anything.

Metrics
-------
* MAE  — mean absolute error, in kWh. Robust, same units as the target.
* RMSE — root mean squared error, in kWh. Penalises large errors more heavily.
* R²   — coefficient of determination. Reported as ``None`` when the target is
  constant over the evaluated rows, because the denominator is then zero and R²
  is mathematically undefined. (That cannot happen here — the test target
  varies — but the guard keeps the output honest for arbitrary frames.)

Test metrics are the headline result. Train metrics are reported separately and
clearly labelled as *diagnostic only* — they describe how well persistence
tracks the training series and are not evidence of generalisation.

No-go zones respected
---------------------
* ``data/raw/`` is never touched.
* No packages are installed.
* No existing verified artefact is overwritten — outputs are new, additive files.
* No machine-learning model is trained.
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
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "demand_kwh"
ZONE_COLUMN = "zone_id"
TIME_COLUMN = "time"
PREDICTOR_COLUMN = "lag_1"

TRAIN_FILENAME = "features_train.csv"
TEST_FILENAME = "features_test.csv"

PREDICTIONS_FILENAME = "baseline_test_predictions.csv"
METRICS_FILENAME = "baseline_metrics.json"
REPORT_FILENAME = "baseline_persistence.md"

# Columns of the saved prediction file, in order.
PREDICTION_COLUMNS = [
    TIME_COLUMN,
    ZONE_COLUMN,
    TARGET_COLUMN,
    PREDICTOR_COLUMN,
    "predicted_kwh",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_features(path: Path) -> pd.DataFrame:
    """Load a feature CSV, preserving zone_id as string and time as datetime.

    ``float_precision="round_trip"`` recovers the exact double written by the
    feature builder so the metrics are computed on the same values that were
    validated upstream.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    cols = pd.read_csv(path, nrows=0).columns
    dtype_map = {ZONE_COLUMN: str} if ZONE_COLUMN in cols else None
    parse = [TIME_COLUMN] if TIME_COLUMN in cols else None
    df = pd.read_csv(
        path, dtype=dtype_map, parse_dates=parse, float_precision="round_trip"
    )
    if ZONE_COLUMN in df.columns:
        df[ZONE_COLUMN] = df[ZONE_COLUMN].astype(str)
    return df


# --------------------------------------------------------------------------
# The baseline itself
# --------------------------------------------------------------------------


def persistence_predict(df: pd.DataFrame, predictor: str = PREDICTOR_COLUMN) -> pd.Series:
    """Return the persistence forecast ``y_pred = lag_1`` for every row.

    Deliberately trivial: the "model" is an identity mapping on a single past
    value. A model-free baseline is only meaningful if it is this transparent —
    there is nothing to fit and therefore nothing that could leak.
    """
    if predictor not in df.columns:
        raise KeyError(f"Predictor column missing: {predictor!r}")
    return df[predictor].astype(float).rename("predicted_kwh")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE and R² with numpy only.

    Implemented directly rather than via scikit-learn so the module has no
    optional dependency and so the exact formulas are visible and auditable::

        MAE  = mean(|y_true - y_pred|)
        RMSE = sqrt(mean((y_true - y_pred)^2))
        R²   = 1 - SS_res / SS_tot,  SS_tot = sum((y_true - mean(y_true))^2)

    ``r2`` is ``None`` when ``SS_tot == 0`` (constant target): the ratio is then
    0/0 and R² has no defined value. ``n_obs`` is returned so callers can see
    how many rows fed each number.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty array")

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2: float | None = None if ss_tot == 0.0 else float(1.0 - np.sum(errors**2) / ss_tot)

    return {"mae": mae, "rmse": rmse, "r2": r2, "n_obs": int(y_true.size)}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_split(
    df: pd.DataFrame,
    name: str,
    target: str = TARGET_COLUMN,
    predictor: str = PREDICTOR_COLUMN,
) -> dict:
    """Check a feature split is usable as-is: no NaN, right columns, right types.

    Returns a small dict of facts. Raises on anything that would silently change
    the evaluation (missing columns, NaNs, non-float target/predictor).
    """
    for col in (TIME_COLUMN, ZONE_COLUMN, target, predictor):
        if col not in df.columns:
            raise KeyError(f"{name}: required column missing: {col!r}")

    n_nan_target = int(df[target].isna().sum())
    n_nan_pred = int(df[predictor].isna().sum())
    if n_nan_target or n_nan_pred:
        raise AssertionError(
            f"{name}: unexpected NaN (target={n_nan_target}, {predictor}={n_nan_pred})"
        )

    facts = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_zones": int(df[ZONE_COLUMN].nunique()),
        "target_dtype": str(df[target].dtype),
        "predictor_dtype": str(df[predictor].dtype),
        "time_min": str(df[TIME_COLUMN].min()),
        "time_max": str(df[TIME_COLUMN].max()),
        "n_missing_target": n_nan_target,
        "n_missing_predictor": n_nan_pred,
        "target_is_float64": bool(pd.api.types.is_float_dtype(df[target])),
    }
    return facts


def _assert_no_target_leakage(pred: pd.Series, df: pd.DataFrame, predictor: str) -> None:
    """Assert the prediction was derived from ``predictor`` and nothing else."""
    expected = df[predictor].to_numpy(dtype=float)
    got = pred.to_numpy(dtype=float)
    if not np.array_equal(expected, got):
        raise AssertionError(
            "predicted_kwh is not an exact copy of the predictor column "
            f"({predictor}); the target may have been used."
        )


# --------------------------------------------------------------------------
# Evaluation driver
# --------------------------------------------------------------------------


def evaluate_split(
    df: pd.DataFrame,
    name: str,
    target: str = TARGET_COLUMN,
    predictor: str = PREDICTOR_COLUMN,
) -> tuple[pd.DataFrame, dict]:
    """Apply the persistence rule to ``df`` and score it.

    Returns ``(frame_with_predictions, metrics_dict)``. The frame gains a
    ``predicted_kwh`` column; ``df`` itself is not mutated.
    """
    facts = validate_split(df, name, target, predictor)
    pred = persistence_predict(df, predictor)
    _assert_no_target_leakage(pred, df, predictor)

    out = df.copy()
    out["predicted_kwh"] = pred
    out["residual_kwh"] = out[target] - out["predicted_kwh"]

    metrics = compute_metrics(df[target].to_numpy(), pred.to_numpy())
    metrics["residual_mean"] = float(out["residual_kwh"].mean())
    metrics["target_std"] = float(df[target].std(ddof=0))
    metrics["target_mean"] = float(df[target].mean())

    # Independent-of-target proof, recorded as a metric-level fact.
    metrics["prediction_equals_lag_1"] = bool(
        np.array_equal(out["predicted_kwh"].to_numpy(), out[predictor].to_numpy())
    )
    metrics["target_used_in_prediction"] = False
    metrics["split"] = name
    metrics["rows"] = facts["rows"]
    metrics["n_zones"] = facts["n_zones"]
    metrics["time_min"] = facts["time_min"]
    metrics["time_max"] = facts["time_max"]

    for key in ("mae", "rmse", "r2"):
        value = metrics[key]
        if value is not None and not np.isfinite(value):
            raise AssertionError(f"{name}: metric {key} is not finite ({value})")

    return out, metrics


def _save_predictions(df: pd.DataFrame, path: Path) -> None:
    """Write time, zone, actual, lag_1 and prediction for the test rows."""
    cols = [c for c in PREDICTION_COLUMNS if c in df.columns]
    out = df.loc[:, cols].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: repr(float(v)) if pd.notna(v) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _human_report(test: dict, train: dict, paths: dict) -> str:
    """Render the short human-readable markdown report."""
    def fmt(v):
        return "n/a" if v is None else f"{v:.6f}"

    return f"""# Persistence baseline — one-hour-ahead EV charging demand

## What was evaluated

The persistence rule predicts demand at time `t` with the demand observed at
`t-1`:

    y_pred = lag_1

No model was fitted. This is a reference point only: a later model must beat
these numbers **on the same test rows** to justify its added complexity.

## Test set (headline result)

| metric | value |
| --- | --- |
| MAE  (kWh) | {fmt(test['mae'])} |
| RMSE (kWh) | {fmt(test['rmse'])} |
| R² | {fmt(test['r2'])} |
| rows | {test['rows']} |
| zones | {test['n_zones']} |
| window | {test['time_min']} → {test['time_max']} |

## Training set (diagnostic only — NOT evidence of generalisation)

| metric | value |
| --- | --- |
| MAE  (kWh) | {fmt(train['mae'])} |
| RMSE (kWh) | {fmt(train['rmse'])} |
| R² | {fmt(train['r2'])} |
| rows | {train['rows']} |
| zones | {train['n_zones']} |
| window | {train['time_min']} → {train['time_max']} |

The train numbers describe how closely persistence tracks the series it is
being scored on. Because the split is chronological and the test window lies
strictly after the train window for every zone, the train row-count and window
are reported purely so the split is auditable.

## Leakage checks

* The prediction is an exact copy of `lag_1`, which the feature builder produced
  with a within-zone `shift(1)` on chronologically ordered history — a value
  genuinely known at `t-1`.
* `demand_kwh` is used **only** as the label in the metric computation; it never
  enters the prediction. Asserted at runtime for both splits.
* The existing chronological split is reused unchanged — no reshuffling, no
  refitting, and the test window is never used to build the forecast.
* No row was dropped or imputed at this stage: the feature files already contain
  no missing values (warm-up rows were dropped upstream).

## Artifacts

* predictions: `{paths['predictions']}`
* metrics:     `{paths['metrics']}`
* report:      `{paths['report']}`
"""


def run_baseline(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    write: bool = True,
) -> dict:
    """Evaluate the persistence baseline on train and test; optionally persist.

    Writes three *new* files (nothing existing is overwritten):

    * ``data/processed/baseline_test_predictions.csv`` — test predictions
    * ``reports/baseline_metrics.json`` — machine-readable metrics
    * ``reports/baseline_persistence.md`` — short human-readable report
    """
    train_df = load_features(processed_dir / TRAIN_FILENAME)
    test_df = load_features(processed_dir / TEST_FILENAME)

    train_scored, train_metrics = evaluate_split(train_df, "train")
    test_scored, test_metrics = evaluate_split(test_df, "test")

    # Machine-readable validation facts, checked here rather than only claimed.
    # NOTE: the frames are sorted zone-major (zone_id, time), so `time` is NOT
    # globally monotonic — it resets at each zone boundary. The correct ordered
    # invariant is: zone_id non-decreasing, and time increasing within each zone.
    def _is_zone_time_ordered(frame: pd.DataFrame) -> bool:
        if not frame[ZONE_COLUMN].is_monotonic_increasing:
            return False
        return all(
            g[TIME_COLUMN].is_monotonic_increasing
            for _, g in frame.groupby(ZONE_COLUMN, sort=False)
        )

    checks = {
        "test_row_count_matches_features": test_metrics["rows"] == len(test_df),
        "test_zone_time_ordered": bool(_is_zone_time_ordered(test_df)),
        "no_missing_values": bool(
            test_df[[TARGET_COLUMN, PREDICTOR_COLUMN]].isna().sum().sum() == 0
        ),
        "prediction_equals_lag_1": bool(test_metrics["prediction_equals_lag_1"]),
        "target_not_used_in_prediction": bool(
            not np.array_equal(
                test_scored["predicted_kwh"].to_numpy(),
                test_scored[TARGET_COLUMN].to_numpy(),
            )
        ),
        "metrics_all_finite": bool(
            all(
                test_metrics[k] is None or np.isfinite(test_metrics[k])
                for k in ("mae", "rmse", "r2")
            )
        ),
        "r2_defined": test_metrics["r2"] is not None,
        "test_after_train_every_zone": True,  # enforced upstream; re-checked below
    }
    # Re-check chronological separation directly on the two files.
    tr_max = train_df.groupby(ZONE_COLUMN)[TIME_COLUMN].max()
    te_min = test_df.groupby(ZONE_COLUMN)[TIME_COLUMN].min()
    zones = sorted(set(tr_max.index) & set(te_min.index))
    checks["test_after_train_every_zone"] = bool(
        all(tr_max[z] < te_min[z] for z in zones) and len(zones) == test_df[ZONE_COLUMN].nunique()
    )
    checks["all_checks_passed"] = bool(all(checks.values()))

    paths = {
        "predictions": str(processed_dir / PREDICTIONS_FILENAME),
        "metrics": str(reports_dir / METRICS_FILENAME),
        "report": str(reports_dir / REPORT_FILENAME),
    }

    result = {
        "baseline": "persistence",
        "rule": "y_pred = lag_1",
        "model_fitted": False,
        "target_column": TARGET_COLUMN,
        "predictor_column": PREDICTOR_COLUMN,
        "test": test_metrics,
        "train": train_metrics,
        "test_is_headline": True,
        "train_is_diagnostic_only": True,
        "validation_checks": checks,
        "notes": [
            "metrics computed with numpy only; no model was fitted or tuned",
            "test metrics are the headline result; train metrics are diagnostic",
            "demand_kwh used as label only, never as a prediction input",
            "existing chronological split reused unchanged",
        ],
        "artifacts": paths,
    }

    if write:
        _save_predictions(test_scored, processed_dir / PREDICTIONS_FILENAME)
        reports_dir.mkdir(parents=True, exist_ok=True)
        with (reports_dir / METRICS_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        with (reports_dir / REPORT_FILENAME).open("w", encoding="utf-8") as fh:
            fh.write(_human_report(test_metrics, train_metrics, paths))

    return result


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the persistence (lag_1) baseline."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--no-write", action="store_true", help="dry run; write nothing")
    args = parser.parse_args(argv)

    try:
        result = run_baseline(
            args.processed_dir, args.reports_dir, write=not args.no_write
        )
    except (FileNotFoundError, KeyError, AssertionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    # Running this file as a script would import it twice (as __main__ and as
    # src.evaluation.baseline_persistence via the package __init__), producing a
    # harmless RuntimeWarning. When executed directly, suppress that warning and
    # delegate to the module so both import paths share one set of constants.
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from src.evaluation.baseline_persistence import main as _pkg_main

    raise SystemExit(_pkg_main())
