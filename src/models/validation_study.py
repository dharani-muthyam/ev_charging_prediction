"""Time-based validation, bounded model selection, and a final test evaluation.

Why a separate module
---------------------
``src/models/train_models.py`` is the *verified reference* pipeline: it trains
Linear Regression and Random Forest with a fixed configuration on the existing
chronological split and writes the reports the dashboard reads. That behaviour
is preserved untouched. This module is the *experiment* layer:

    reference train period                       final test period
    |------------------------------------------| |------------------|
    |             fit search period            | |  held-out (hol)",
    |                                   | valid| |
                                        train    valid
                                        80%      20%

1. The **final test period is never touched** during model selection. It is
   only used once, at the very end, to score the single candidate chosen on
   validation.
2. A validation fold is carved from the **end** of each zone's reference
   training rows (default: the most recent 20%, per zone, by position). This
   keeps the per-zone chronological ordering and, because lag features were
   already built upstream with within-zone shifts, no future observation can
   enter a training feature: rows are only ever *removed* from the tail.
3. Candidate configurations are compared on validation only (MAE, RMSE, R²).
4. The best candidate is then refitted on the full reference training rows and
   scored once on the final test rows; the result is reported **separately**
   from validation numbers and next to the recorded reference results.

Bounded by design
-----------------
The search grid is deliberately small (a handful of Random Forest settings plus
one HistGradientBoosting setting), each grid point gets a bounded number of trees
and ``max_iter``, and the module supports sampling for quick smoke runs. Nothing
here writes to ``models/`` or overwrites the reference reports: experiment
outputs go to ``reports/experiments/`` and ``results/experiments/``.

Usage
-----
    python -m src.models.validation_study                # full study
    python -m src.models.validation_study --quick        # sampled smoke run
    python -m src.models.validation_study --no-write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from src.evaluation.baseline_persistence import compute_metrics
from src.models.train_models import (
    PREDICTOR_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TIME_COLUMN,
    ZONE_COLUMN,
    ZONE_PREDICTOR_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "experiments"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

TRAIN_FILENAME = "features_train_zone.csv"
TEST_FILENAME = "features_test_zone.csv"

VALIDATION_FRACTION = 0.2

# Small, bounded grids. Each entry is a full estimator configuration so the
# comparison is explicit rather than an implicit search-space product.
CANDIDATES: tuple[dict, ...] = (
    {
        "name": "rf_reference",
        "label": "Random Forest (reference: 100 trees, depth 20, leaf 20)",
        "family": "rf",
        "params": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 20},
        "predictors": "base",
    },
    {
        "name": "rf_leaf50_depth16",
        "label": "Random Forest (150 trees, depth 16, leaf 50)",
        "family": "rf",
        "params": {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 50},
        "predictors": "base",
    },
    {
        "name": "rf_leaf100_depth24",
        "label": "Random Forest (150 trees, depth 24, leaf 100)",
        "family": "rf",
        "params": {"n_estimators": 150, "max_depth": 24, "min_samples_leaf": 100},
        "predictors": "base",
    },
    {
        "name": "rf_zone_features",
        "label": "Random Forest (reference params + static zone features)",
        "family": "rf",
        "params": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 20},
        "predictors": "zone",
    },
    {
        "name": "hgb_leaf100_lr005",
        "label": "HistGradientBoosting (400 iters, lr 0.05, leaf 100)",
        "family": "hgb",
        "params": {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 100},
        "predictors": "base",
    },
    {
        "name": "hgb_zone_features",
        "label": "HistGradientBoosting (400 iters, lr 0.05, leaf 100 + zone features)",
        "family": "hgb",
        "params": {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 100},
        "predictors": "zone",
    },
)


# --------------------------------------------------------------------------
# Loading and splitting
# --------------------------------------------------------------------------


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the pipeline up to "
            "`python -m src.features.zone_features` first."
        )
    df = pd.read_csv(
        path, dtype={ZONE_COLUMN: str}, parse_dates=[TIME_COLUMN], float_precision="round_trip"
    )
    df[ZONE_COLUMN] = df[ZONE_COLUMN].astype(str)
    return df


def temporal_train_valid_split(
    train_df: pd.DataFrame, validation_fraction: float = VALIDATION_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split each zone's reference-training rows into search-train / validation.

    The validation fold is the **most recent** ``validation_fraction`` of each
    zone's rows, so validation sits strictly after training for every zone.
    No row is duplicated and no row from the final test file participates.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    search_parts, valid_parts = [], []
    for _, g in train_df.groupby(ZONE_COLUMN, sort=True):
        g = g.sort_values(TIME_COLUMN, kind="mergesort")
        n_valid = int(len(g) * validation_fraction)
        n_valid = min(max(n_valid, 1), len(g) - 1)
        split = len(g) - n_valid
        search_parts.append(g.iloc[:split])
        valid_parts.append(g.iloc[split:])

    search = pd.concat(search_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)

    tr_max = search.groupby(ZONE_COLUMN)[TIME_COLUMN].max()
    va_min = valid.groupby(ZONE_COLUMN)[TIME_COLUMN].min()
    zones = sorted(set(tr_max.index) & set(va_min.index))
    bad = [z for z in zones if not tr_max[z] < va_min[z]]
    if bad:
        raise AssertionError(f"validation not strictly after training for zones: {bad[:5]}")

    return search, valid, {
        "train_rows": int(len(search)),
        "valid_rows": int(len(valid)),
        "n_zones": int(len(zones)),
        "train_time_min": str(search[TIME_COLUMN].min()),
        "train_time_max": str(search[TIME_COLUMN].max()),
        "valid_time_min": str(valid[TIME_COLUMN].min()),
        "valid_time_max": str(valid[TIME_COLUMN].max()),
        "validation_fraction": validation_fraction,
        "chronological_per_zone": True,
    }


# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------


def build_candidate(spec: dict):
    if spec["family"] == "rf":
        return RandomForestRegressor(
            random_state=RANDOM_SEED, n_jobs=-1, **spec["params"]
        )
    if spec["family"] == "hgb":
        return HistGradientBoostingRegressor(random_state=RANDOM_SEED, **spec["params"])
    raise ValueError(f"unknown family {spec['family']!r}")


def predictors_for(spec: dict) -> tuple[str, ...]:
    return ZONE_PREDICTOR_COLUMNS if spec["predictors"] == "zone" else PREDICTOR_COLUMNS


def _subset(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Optional bounded sample for smoke runs (chronology is not used by these fits)."""
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_values(
        [ZONE_COLUMN, TIME_COLUMN], kind="mergesort"
    )


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = compute_metrics(y_true, y_pred)
    return {"mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"], "rows": int(m["n_obs"])}


def evaluate_on_validation(
    spec: dict,
    search: pd.DataFrame,
    valid: pd.DataFrame,
    max_rows: int | None = None,
) -> dict:
    predictors = list(predictors_for(spec))
    fit_rows = _subset(search, max_rows, RANDOM_SEED)
    model = build_candidate(spec)
    started = time.perf_counter()
    model.fit(fit_rows[predictors], fit_rows[TARGET_COLUMN].to_numpy(dtype=float))
    seconds = time.perf_counter() - started

    pred = np.asarray(model.predict(valid[predictors]), dtype=float)
    if not np.all(np.isfinite(pred)):
        raise AssertionError(f"{spec['name']}: non-finite validation predictions")
    return {
        "name": spec["name"],
        "label": spec["label"],
        "family": spec["family"],
        "params": spec["params"],
        "predictors_mode": spec["predictors"],
        "n_predictors": len(predictors),
        "fit_rows": int(len(fit_rows)),
        "valid": _score(valid[TARGET_COLUMN].to_numpy(dtype=float), pred),
        "fit_seconds": round(seconds, 2),
    }


# --------------------------------------------------------------------------
# Study driver
# --------------------------------------------------------------------------


def run_study(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    validation_fraction: float = VALIDATION_FRACTION,
    write: bool = True,
    max_fit_rows: int | None = None,
    candidates: tuple[dict, ...] = CANDIDATES,
) -> dict:
    train_df = load_frame(processed_dir / TRAIN_FILENAME)
    test_df = load_frame(processed_dir / TEST_FILENAME)

    search, valid, split_info = temporal_train_valid_split(train_df, validation_fraction)

    validation_results = [
        evaluate_on_validation(spec, search, valid, max_fit_rows) for spec in candidates
    ]
    # Selection rule: lowest validation RMSE (declared up front, not chosen
    # after seeing the test set). R² would rank identically here because both
    # are measured on the same validation rows, but RMSE is the primary metric.
    winner = min(validation_results, key=lambda r: r["valid"]["rmse"])
    winner_spec = next(s for s in candidates if s["name"] == winner["name"])

    # ---- the final test set is used exactly once, for the chosen candidate ----
    predictors = list(predictors_for(winner_spec))
    final_fit_rows = _subset(train_df, max_fit_rows, RANDOM_SEED)
    started = time.perf_counter()
    final_model = build_candidate(winner_spec)
    final_model.fit(final_fit_rows[predictors], final_fit_rows[TARGET_COLUMN].to_numpy(dtype=float))
    fit_seconds = time.perf_counter() - started

    test_pred = np.asarray(final_model.predict(test_df[predictors]), dtype=float)
    if not np.all(np.isfinite(test_pred)):
        raise AssertionError("non-finite predictions on the final test set")
    test_metrics = _score(test_df[TARGET_COLUMN].to_numpy(dtype=float), test_pred)
    test_metrics["n_zones"] = int(test_df[ZONE_COLUMN].nunique())
    test_metrics["time_min"] = str(test_df[TIME_COLUMN].min())
    test_metrics["time_max"] = str(test_df[TIME_COLUMN].max())

    reference = load_reference_metrics(PROJECT_ROOT)

    results = {
        "design": {
            "validation_fold": split_info,
            "final_test": {
                "rows": int(len(test_df)),
                "n_zones": int(test_df[ZONE_COLUMN].nunique()),
                "time_min": str(test_df[TIME_COLUMN].min()),
                "time_max": str(test_df[TIME_COLUMN].max()),
            },
            "selection_rule": "lowest validation RMSE (chosen before viewing test metrics)",
            "test_used_for_selection": False,
            "random_seed": RANDOM_SEED,
            "max_fit_rows": max_fit_rows,
            "quick_mode": max_fit_rows is not None,
        },
        "validation_results": validation_results,
        "selected_candidate": {
            "name": winner_spec["name"],
            "label": winner_spec["label"],
            "family": winner_spec["family"],
            "params": winner_spec["params"],
            "predictors": predictors,
            "validation": winner["valid"],
            "refit_rows": int(len(final_fit_rows)),
            "refit_seconds": round(fit_seconds, 2),
        },
        "final_test": test_metrics,
        "reference": reference,
        "verdict": _verdict(test_metrics, reference, winner, validation_results),
    }

    if write:
        reports_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        with (reports_dir / "validation_study.json").open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

        table_rows = []
        for r in validation_results:
            table_rows.append(
                {
                    "stage": "validation",
                    "candidate": r["name"],
                    "label": r["label"],
                    "mae": r["valid"]["mae"],
                    "rmse": r["valid"]["rmse"],
                    "r2": r["valid"]["r2"],
                    "rows": r["valid"]["rows"],
                    "fit_seconds": r["fit_seconds"],
                }
            )
        for name, m in reference.get("models", {}).items():
            table_rows.append(
                {
                    "stage": "reference_test",
                    "candidate": name,
                    "label": m.get("label", name),
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                    "rows": m.get("rows"),
                    "fit_seconds": None,
                }
            )
        table_rows.append(
            {
                "stage": "final_test",
                "candidate": winner_spec["name"],
                "label": winner_spec["label"],
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "rows": test_metrics["rows"],
                "fit_seconds": fit_seconds,
            }
        )
        pd.DataFrame(table_rows).to_csv(
            results_dir / "validation_study.csv", index=False
        )
        with (reports_dir / "validation_study.md").open("w", encoding="utf-8") as fh:
            fh.write(_markdown(results))
        results["artifacts"] = {
            "json": str(reports_dir / "validation_study.json"),
            "markdown": str(reports_dir / "validation_study.md"),
            "table": str(results_dir / "validation_study.csv"),
        }

    return results


def load_reference_metrics(root: Path) -> dict:
    """Read the recorded reference test metrics (read-only)."""
    path = root / "reports" / "model_metrics.json"
    artifact = root / "reports" / "baseline_metrics.json"
    out: dict = {"models": {}, "baseline": None}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            mm = json.load(fh)
        for key, m in mm.get("models", {}).items():
            out["models"][key] = {"label": m.get("label", key), **m["test"]}
    if artifact.exists():
        with artifact.open(encoding="utf-8") as fh:
            bm = json.load(fh)
        out["baseline"] = {k: bm["test"][k] for k in ("mae", "rmse", "r2", "rows")}
    return out


def _verdict(test: dict, reference: dict, winner: dict, all_results: list[dict]) -> dict:
    """Honest, explicit comparison — no improvement claim without a valid basis."""
    ref = reference.get("models", {})
    best_ref = min(ref.items(), key=lambda kv: kv[1]["rmse"]) if ref else None
    if best_ref is None:
        return {"note": "reference metrics unavailable; no comparison made"}
    name, m = best_ref
    beats = (
        test["mae"] < m["mae"] and test["rmse"] < m["rmse"] and test["r2"] > m["r2"]
    )
    return {
        "reference_model": name,
        "reference_test": {k: m[k] for k in ("mae", "rmse", "r2")},
        "selected_test": {k: test[k] for k in ("mae", "rmse", "r2")},
        "beats_reference_on_all_three": bool(beats),
        "delta_mae": float(test["mae"] - m["mae"]),
        "delta_rmse": float(test["rmse"] - m["rmse"]),
        "delta_r2": float(test["r2"] - m["r2"]),
        "same_test_rows": bool(int(test["rows"]) == int(m.get("rows", -1))),
        "selected_was_best_on_validation": bool(
            winner["valid"]["rmse"] == min(r["valid"]["rmse"] for r in all_results)
        ),
        "caveat": (
            "validation ranking does not guarantee a test-set improvement; the "
            "verdict above is based on the measured final-test numbers only"
        ),
    }


def _markdown(r: dict) -> str:
    d = r["design"]
    v = d["validation_fold"]
    lines = [
        "# Time-based validation study (experiment layer)",
        "",
        "The existing reference models and reports were **not** modified by this study.",
        "",
        "## Design",
        "",
        f"* Search-train rows: {v['train_rows']} ({v['train_time_min']} → {v['train_time_max']})",
        f"* Validation rows: {v['valid_rows']} ({v['valid_time_min']} → {v['valid_time_max']})",
        f"  — most recent {v['validation_fraction']:.0%} of each zone's reference-training rows",
        f"* Final test rows: {d['final_test']['rows']} ({d['final_test']['time_min']} → {d['final_test']['time_max']})",
        "* Selection rule: lowest validation RMSE, fixed **before** the test set was scored",
        f"* Random seed: {d['random_seed']}"
        + (f" | quick mode, max fit rows: {d['max_fit_rows']}" if d["max_fit_rows"] else ""),
        "",
        "## Validation comparison",
        "",
        "| candidate | MAE (kWh) | RMSE (kWh) | R² | fit s |",
        "| --- | --- | --- | --- | --- |",
    ]
    for res in sorted(r["validation_results"], key=lambda x: x["valid"]["rmse"]):
        m = res["valid"]
        lines.append(
            f"| {res['label']} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | {res['fit_seconds']:.1f} |"
        )
    t = r["final_test"]
    sel = r["selected_candidate"]
    lines += [
        "",
        "## Final test result (single candidate, scored once)",
        "",
        f"**Selected on validation:** {sel['label']}",
        "",
        "| set | MAE (kWh) | RMSE (kWh) | R² |",
        "| --- | --- | --- | --- |",
        f"| validation | {sel['validation']['mae']:.4f} | {sel['validation']['rmse']:.4f} | {sel['validation']['r2']:.4f} |",
        f"| final test | {t['mae']:.4f} | {t['rmse']:.4f} | {t['r2']:.4f} |",
        "",
        "## Versus the reference results (recorded, unmodified)",
        "",
    ]
    verdict = r["verdict"]
    if "reference_test" in verdict:
        ref = verdict["reference_test"]
        lines += [
            "| model | MAE (kWh) | RMSE (kWh) | R² | same test rows |",
            "| --- | --- | --- | --- | --- |",
            f"| reference {verdict['reference_model']} | {ref['mae']:.4f} | {ref['rmse']:.4f} | {ref['r2']:.4f} | yes |",
            f"| selected candidate | {t['mae']:.4f} | {t['rmse']:.4f} | {t['r2']:.4f} | {verdict['same_test_rows']} |",
            "",
            f"Beats the reference on **all three** test metrics: "
            f"**{'YES' if verdict['beats_reference_on_all_three'] else 'NO'}** "
            f"(ΔMAE {verdict['delta_mae']:+.4f}, ΔRMSE {verdict['delta_rmse']:+.4f}, "
            f"ΔR² {verdict['delta_r2']:+.4f}).",
            "",
            f"> {verdict['caveat']}.",
        ]
    else:
        lines.append(verdict.get("note", "no comparison available"))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time-based validation study.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--quick", action="store_true", help="sample fit rows (smoke run)")
    parser.add_argument("--max-fit-rows", type=int, default=None)
    parser.add_argument("--no-write", action="store_true", help="dry run; write nothing")
    args = parser.parse_args(argv)

    max_rows = args.max_fit_rows if args.max_fit_rows is not None else (150_000 if args.quick else None)

    try:
        results = run_study(
            args.processed_dir,
            args.reports_dir,
            args.results_dir,
            validation_fraction=args.validation_fraction,
            write=not args.no_write,
            max_fit_rows=max_rows,
        )
    except (FileNotFoundError, KeyError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
