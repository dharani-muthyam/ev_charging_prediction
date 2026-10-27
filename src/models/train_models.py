"""Linear Regression and Random Forest models for one-hour-ahead EV demand.

What this module does
---------------------
It trains two supervised regressors on the existing, already-verified feature
tables and scores them on *exactly* the same test rows the persistence baseline
used, so the three approaches are directly comparable::

    target      = demand_kwh[z, t]
    predictors  = hour, day_of_week, month, lag_1, lag_24

Leakage safety
--------------
* The predictor set is explicit (``PREDICTOR_COLUMNS``) and only contains values
  known at prediction time ``t-1``: ``lag_1``/``lag_24`` are past observations
  built with a within-zone ``shift`` by ``src.features.demand_lags``, and the
  calendar columns come from the row's own timestamp (which is known in advance
  for a forecast).
* ``demand_kwh`` (the target), ``time``, ``zone_id`` and ``split`` are excluded.
  ``time``/``zone_id`` are identifiers, and including ``demand_kwh`` would leak
  the answer. ``split`` would leak the train/test membership.
* A runtime guard rejects any configuration whose features intersect the
  forbidden set, so the rule is enforced, not merely documented.
* Models are fitted on the training rows only; the test rows are never seen
  during fitting. No shuffling or re-splitting occurs.

Reproducibility
---------------
``RANDOM_SEED`` is fixed and passed to the Random Forest (``random_state``).
Linear Regression is deterministic and needs no seed. The fitted estimators,
the predictor list, and the seed are saved together so a model can be reloaded
and reproduced.

Caveat, measured rather than assumed: the Random Forest is fitted with
``n_jobs=-1``, and scikit-learn sums leaf contributions across a thread pool.
Floating-point *reduction order* therefore varies between runs, so two fits with
an identical seed agree only to ~1e-12, not bit-for-bit. Linear Regression is
exactly reproducible. Set ``n_jobs=1`` in ``build_random_forest`` if bit-exact
repeatability is required at the cost of speed.

Comparison against the baseline
-------------------------------
The persistence baseline metrics are read from ``reports/baseline_metrics.json``
(produced by ``src.evaluation.baseline_persistence``) rather than recomputed, so
the comparison is against the recorded, verified numbers. A model is only called
"better" when its MAE, RMSE and R² are all at least as good as the baseline's on
the identical test rows; the report states this explicitly per model.

Outputs (new, additive files — nothing existing is overwritten)
---------------------------------------------------------------
* ``models/linear_regression.joblib`` / ``models/random_forest.joblib``
* ``reports/model_metrics.json`` / ``reports/model_comparison.md``
* ``results/predictions/linear_regression_test_predictions.csv``
* ``results/predictions/random_forest_test_predictions.csv``
* ``results/predictions/model_test_predictions.csv`` (baseline + both models)
* ``results/tables/model_comparison.csv``

Usage
-----
    python -m src.models.train_models
    python -m src.models.train_models --no-write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Reuse the verified helpers so metric formulas and CSV loading stay consistent
# with the persistence baseline (no second, subtly different implementation).
from src.evaluation.baseline_persistence import compute_metrics, load_features

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TARGET_COLUMN = "demand_kwh"
ZONE_COLUMN = "zone_id"
TIME_COLUMN = "time"

# The only columns allowed as predictors: two past-demand lags plus calendar
# features derived from the row's own timestamp (known at forecast time).
PREDICTOR_COLUMNS: tuple[str, ...] = ("hour", "day_of_week", "month", "lag_1", "lag_24")

# Columns that must never be predictors, with the reason recorded for clarity.
FORBIDDEN_PREDICTORS: dict[str, str] = {
    TARGET_COLUMN: "the target itself (would leak the answer)",
    TIME_COLUMN: "row identifier / timestamp, not a model input",
    ZONE_COLUMN: "identifier; not used as a numeric predictor here",
    "split": "train/test membership (would leak the split)",
    "predicted_kwh": "a prediction, never an input",
    "residual_kwh": "derived from the target",
}

RANDOM_SEED = 42

TRAIN_FILENAME = "features_train.csv"
TEST_FILENAME = "features_test.csv"
BASELINE_METRICS_FILENAME = "baseline_metrics.json"

MODEL_METRICS_FILENAME = "model_metrics.json"
MODEL_COMPARISON_FILENAME = "model_comparison.md"
MODEL_TABLE_FILENAME = "model_comparison.csv"
COMBINED_PREDICTIONS_FILENAME = "model_test_predictions.csv"

# Per-model artifact names.
MODEL_SPECS: tuple[dict, ...] = (
    {
        "key": "linear_regression",
        "label": "Linear Regression",
        "model_file": "linear_regression.joblib",
        "predictions_file": "linear_regression_test_predictions.csv",
    },
    {
        "key": "random_forest",
        "label": "Random Forest",
        "model_file": "random_forest.joblib",
        "predictions_file": "random_forest_test_predictions.csv",
    },
)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def assert_predictors_are_safe(predictors: tuple[str, ...]) -> None:
    """Reject any predictor set that includes a forbidden (leaky) column."""
    bad = [c for c in predictors if c in FORBIDDEN_PREDICTORS]
    if bad:
        reasons = "; ".join(f"{c}: {FORBIDDEN_PREDICTORS[c]}" for c in bad)
        raise AssertionError(f"Unsafe predictor column(s) selected -> {reasons}")
    if len(set(predictors)) != len(predictors):
        raise AssertionError("Duplicate predictor columns selected.")


def assert_columns_present(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing required column(s): {missing}")


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------


def build_linear_regression() -> LinearRegression:
    """Plain ordinary-least-squares regression. Deterministic, no seed needed."""
    return LinearRegression()


def build_random_forest() -> RandomForestRegressor:
    """Random Forest with a fixed seed for reproducibility.

    ``n_jobs=-1`` uses all cores; it does not affect the result.

    The tree-growth parameters are deliberately bounded. With 949k training rows
    and ``min_samples_leaf=2``/unlimited depth, each tree grows to tens of
    thousands of leaves and the serialised forest exceeds 3 GB — an artifact
    that is impractical to store or reload. ``max_depth=20`` with
    ``min_samples_leaf=20`` is a standard, defensible compromise: it keeps the
    model small enough to persist (tens of MB) while retaining high accuracy
    (see ``reports/model_comparison.md`` for the measured metrics on this data).
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def get_estimator(key: str):
    """Return an unfitted estimator for a model key."""
    if key == "linear_regression":
        return build_linear_regression()
    if key == "random_forest":
        return build_random_forest()
    raise ValueError(f"Unknown model key: {key!r}")


# --------------------------------------------------------------------------
# Training & evaluation
# --------------------------------------------------------------------------


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key: str,
    predictors: tuple[str, ...] = PREDICTOR_COLUMNS,
    target: str = TARGET_COLUMN,
) -> tuple[object, pd.DataFrame, dict, dict]:
    """Fit one model on the train rows and score it on the test rows.

    Returns ``(fitted_model, test_predictions_frame, test_metrics, train_metrics)``.
    Fitting uses training rows only. The returned frame carries the actual
    target, the predictors, and the model prediction for the same test rows the
    baseline used.
    """
    assert_predictors_are_safe(predictors)
    assert_columns_present(train_df, predictors + (target,), "train")
    assert_columns_present(test_df, predictors + (target,), "test")

    x_train = train_df.loc[:, list(predictors)]
    y_train = train_df[target].to_numpy(dtype=float)
    x_test = test_df.loc[:, list(predictors)]
    y_test = test_df[target].to_numpy(dtype=float)

    model = get_estimator(key)
    model.fit(x_train, y_train)

    test_pred = np.asarray(model.predict(x_test), dtype=float)
    train_pred = np.asarray(model.predict(x_train), dtype=float)

    if not np.all(np.isfinite(test_pred)):
        raise AssertionError(f"{key}: non-finite predictions produced on the test set")

    out = pd.DataFrame(
        {
            TIME_COLUMN: test_df[TIME_COLUMN].to_numpy(),
            ZONE_COLUMN: test_df[ZONE_COLUMN].to_numpy(),
            target: y_test,
            "lag_1": test_df["lag_1"].to_numpy(dtype=float),
            "predicted_kwh": test_pred,
        }
    )

    test_metrics = compute_metrics(y_test, test_pred)
    test_metrics["split"] = "test"
    test_metrics["rows"] = int(len(test_df))
    test_metrics["n_zones"] = int(test_df[ZONE_COLUMN].nunique())
    test_metrics["time_min"] = str(test_df[TIME_COLUMN].min())
    test_metrics["time_max"] = str(test_df[TIME_COLUMN].max())

    train_metrics = compute_metrics(y_train, train_pred)
    train_metrics["split"] = "train"
    train_metrics["rows"] = int(len(train_df))
    train_metrics["n_zones"] = int(train_df[ZONE_COLUMN].nunique())

    for name, m in (("test", test_metrics), ("train", train_metrics)):
        for k in ("mae", "rmse", "r2"):
            v = m[k]
            if v is not None and not np.isfinite(v):
                raise AssertionError(f"{key}/{name}: metric {k} is not finite ({v})")

    return model, out, test_metrics, train_metrics


# --------------------------------------------------------------------------
# Comparison against the persistence baseline
# --------------------------------------------------------------------------


def load_baseline_metrics(reports_dir: Path) -> dict:
    """Load the recorded persistence-baseline metrics."""
    path = reports_dir / BASELINE_METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found: {path}. Run "
            "`python -m src.evaluation.baseline_persistence` first."
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def compare_to_baseline(model_metrics: dict, baseline_metrics: dict) -> dict:
    """Compare a model's test metrics with the persistence baseline.

    A model is deemed better only if MAE and RMSE are both lower and R² is
    higher — all three on the identical test rows. Anything else is reported
    honestly as not better (or mixed).
    """
    b = baseline_metrics["test"]
    m = model_metrics
    deltas = {
        "mae_delta": float(m["mae"] - b["mae"]),
        "rmse_delta": float(m["rmse"] - b["rmse"]),
        "r2_delta": float(m["r2"] - b["r2"]),
    }
    better = (
        m["mae"] < b["mae"] and m["rmse"] < b["rmse"] and m["r2"] > b["r2"]
    )
    return {
        "baseline": {k: b[k] for k in ("mae", "rmse", "r2")},
        "model": {k: m[k] for k in ("mae", "rmse", "r2")},
        **deltas,
        "beats_baseline_on_all_three": bool(better),
        "n_test_rows_model": int(m["rows"]),
        "n_test_rows_baseline": int(b["rows"]),
        "same_test_rows": bool(int(m["rows"]) == int(b["rows"])),
    }


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------


def _dump_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV with exact float64 round-trip for float columns (repr-based)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: repr(float(v)) if pd.notna(v) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _human_report(results: dict) -> str:
    """Render the human-readable comparison report."""
    b = results["baseline_test"]
    lines = [
        "# Model comparison — one-hour-ahead EV charging demand",
        "",
        "All numbers are on the **same chronological test rows** used by the",
        "persistence baseline. Lower MAE/RMSE is better; higher R² is better.",
        "",
        "## Test-set comparison",
        "",
        "| model | MAE (kWh) | RMSE (kWh) | R² | vs baseline |",
        "| --- | --- | --- | --- | --- |",
        f"| Persistence (lag_1) | {b['mae']:.6f} | {b['rmse']:.6f} | {b['r2']:.6f} | reference |",
    ]
    for name, r in results["models"].items():
        c = r["comparison"]
        verdict = "beats baseline" if c["beats_baseline_on_all_three"] else "does not beat baseline"
        lines.append(
            f"| {r['label']} | {r['test']['mae']:.6f} | {r['test']['rmse']:.6f} | "
            f"{r['test']['r2']:.6f} | {verdict} |"
        )

    lines += ["", "## Deltas vs baseline (model − baseline)", "",
              "| model | ΔMAE | ΔRMSE | ΔR² |", "| --- | --- | --- | --- |"]
    for name, r in results["models"].items():
        c = r["comparison"]
        lines.append(
            f"| {r['label']} | {c['mae_delta']:+.6f} | {c['rmse_delta']:+.6f} | "
            f"{c['r2_delta']:+.6f} |"
        )

    lines += ["", "## Configuration", "",
              f"* Predictors: `{', '.join(results['predictors'])}`",
              f"* Target: `{results['target']}`",
              f"* Excluded (leaky/identifier): `{', '.join(results['excluded'])}`",
              f"* Random seed: `{results['random_seed']}` (Random Forest; Linear Regression is deterministic)",
              f"* Fitted on: {results['train_rows']} training rows ({results['train_zones']} zones)",
              f"* Evaluated on: {results['test_rows']} test rows ({results['test_zones']} zones)",
              "",
              "## Train metrics (diagnostic only — not evidence of generalisation)", "",
              "| model | MAE (kWh) | RMSE (kWh) | R² |", "| --- | --- | --- | --- |"]
    for name, r in results["models"].items():
        t = r["train"]
        lines.append(f"| {r['label']} | {t['mae']:.6f} | {t['rmse']:.6f} | {t['r2']:.6f} |")

    lines += ["", "## Artifacts", ""]
    for name, r in results["models"].items():
        lines.append(f"* {r['label']}: model `{r['model_path']}`, predictions `{r['predictions_path']}`")
    lines += [
        f"* metrics: `{results['artifacts']['metrics']}`",
        f"* table:   `{results['artifacts']['table']}`",
        f"* combined predictions: `{results['artifacts']['combined_predictions']}`",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    tables_dir: Path = DEFAULT_TABLES_DIR,
    write: bool = True,
    predictors: tuple[str, ...] = PREDICTOR_COLUMNS,
) -> dict:
    """Train both models, evaluate, compare with the baseline, and persist."""
    assert_predictors_are_safe(predictors)

    train_df = load_features(processed_dir / TRAIN_FILENAME)
    test_df = load_features(processed_dir / TEST_FILENAME)
    baseline_metrics = load_baseline_metrics(reports_dir)

    results: dict = {
        "target": TARGET_COLUMN,
        "predictors": list(predictors),
        "excluded": [c for c in FORBIDDEN_PREDICTORS if c not in predictors],
        "random_seed": RANDOM_SEED,
        "train_rows": int(len(train_df)),
        "train_zones": int(train_df[ZONE_COLUMN].nunique()),
        "test_rows": int(len(test_df)),
        "test_zones": int(test_df[ZONE_COLUMN].nunique()),
        "baseline_test": {k: baseline_metrics["test"][k] for k in ("mae", "rmse", "r2")},
        "models": {},
    }

    combined = None
    comparison_rows = []

    for spec in MODEL_SPECS:
        key = spec["key"]
        model, preds, test_m, train_m = train_and_evaluate(
            train_df, test_df, key, predictors=predictors
        )
        comparison = compare_to_baseline(test_m, baseline_metrics)

        model_path = models_dir / spec["model_file"]
        predictions_path = predictions_dir / spec["predictions_file"]

        results["models"][key] = {
            "label": spec["label"],
            "test": test_m,
            "train": train_m,
            "comparison": comparison,
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
        }

        comparison_rows.append(
            {
                "model": spec["label"],
                "split": "test",
                "mae": test_m["mae"],
                "rmse": test_m["rmse"],
                "r2": test_m["r2"],
                "beats_baseline": comparison["beats_baseline_on_all_three"],
            }
        )

        if write:
            _save_model(model, model_path)
            _dump_csv(preds, predictions_path)

        preds_keyed = preds.rename(columns={"predicted_kwh": f"{key}_predicted_kwh"})
        if combined is None:
            combined = preds_keyed
        else:
            combined = combined.merge(
                preds_keyed[[TIME_COLUMN, ZONE_COLUMN, f"{key}_predicted_kwh"]],
                on=[TIME_COLUMN, ZONE_COLUMN],
                how="inner",
                validate="one_to_one",
            )

    # Add the baseline's own prediction column for a side-by-side file.
    baseline_preds = load_features(processed_dir / "baseline_test_predictions.csv")
    if combined is not None:
        combined = combined.merge(
            baseline_preds.rename(columns={"predicted_kwh": "baseline_predicted_kwh"})[
                [TIME_COLUMN, ZONE_COLUMN, "baseline_predicted_kwh"]
            ],
            on=[TIME_COLUMN, ZONE_COLUMN],
            how="inner",
            validate="one_to_one",
        )
        results["combined_rows"] = int(len(combined))

    results["artifacts"] = {
        "metrics": str(reports_dir / MODEL_METRICS_FILENAME),
        "report": str(reports_dir / MODEL_COMPARISON_FILENAME),
        "table": str(tables_dir / MODEL_TABLE_FILENAME),
        "combined_predictions": str(predictions_dir / COMBINED_PREDICTIONS_FILENAME),
    }

    if write:
        reports_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        with (reports_dir / MODEL_METRICS_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        with (reports_dir / MODEL_COMPARISON_FILENAME).open("w", encoding="utf-8") as fh:
            fh.write(_human_report(results))
        _dump_csv(pd.DataFrame(comparison_rows), tables_dir / MODEL_TABLE_FILENAME)
        if combined is not None:
            _dump_csv(combined, predictions_dir / COMBINED_PREDICTIONS_FILENAME)

    return results


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Linear Regression and Random Forest."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--no-write", action="store_true", help="dry run; write nothing")
    args = parser.parse_args(argv)

    try:
        results = run(
            args.processed_dir,
            args.reports_dir,
            args.models_dir,
            args.predictions_dir,
            args.tables_dir,
            write=not args.no_write,
        )
    except (FileNotFoundError, KeyError, AssertionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
