"""Reproducible evaluation figures from the saved test predictions.

Every figure is computed from files the evaluation pipeline wrote — never from a
re-fit model — so a figure can only show what was actually measured:

    results/predictions/model_test_predictions.csv   (time, zone_id,
        demand_kwh, lag_1, baseline/linear_regression/random_forest predictions)
    results/tables/model_comparison.csv              (test-set MAE/RMSE/R²)

Figures written to ``results/figures/`` (PNG, 120 dpi, modest size):

1. ``fig_actual_vs_predicted.png``  — predicted vs actual, one panel per model
2. ``fig_error_distribution.png``   — residual histograms per model
3. ``fig_model_comparison.png``     — MAE / RMSE / R² bars incl. persistence
4. ``fig_zone_error.png``           — per-zone MAE, worst zones
5. ``fig_high_demand_errors.png``   — error by actual-demand decile

Usage
-----
    python -m src.visualization.plots
    python -m src.visualization.plots --out-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/CI safe; no GUI backend required
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results" / "predictions" / "model_test_predictions.csv"
DEFAULT_COMPARISON = PROJECT_ROOT / "results" / "tables" / "model_comparison.csv"
DEFAULT_BASELINE_METRICS = PROJECT_ROOT / "reports" / "baseline_metrics.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TARGET = "demand_kwh"
ZONE = "zone_id"

# Display order and colours for the three evaluated approaches.
MODEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("baseline_predicted_kwh", "Persistence (lag_1)"),
    ("linear_regression_predicted_kwh", "Linear Regression"),
    ("random_forest_predicted_kwh", "Random Forest"),
)

DPI = 120


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def load_predictions(path: Path = DEFAULT_PREDICTIONS) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"predictions not found: {path}. Run `python -m src.models.train_models`."
        )
    df = pd.read_csv(path, float_precision="round_trip", parse_dates=["time"])
    df[ZONE] = df[ZONE].astype(str)
    return df


def available_models(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(c, label) for c, label in MODEL_COLUMNS if c in df.columns]


# --------------------------------------------------------------------------
# 1. Actual vs predicted
# --------------------------------------------------------------------------


def plot_actual_vs_predicted(df: pd.DataFrame, out_dir: Path) -> Path:
    models = available_models(df)
    if not models:
        raise ValueError("no recognised prediction columns in the predictions file")

    actual = df[TARGET].to_numpy(dtype=float)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    # Subsample for legibility only; all points are used for the reported R².
    limit = 8000
    idx = (
        np.random.default_rng(42).choice(len(df), size=limit, replace=False)
        if len(df) > limit
        else np.arange(len(df))
    )
    lo, hi = float(np.min(actual)), float(np.max(actual))

    for ax, (col, label) in zip(axes[0], models):
        pred = df[col].to_numpy(dtype=float)
        resid = actual - pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        ax.scatter(actual[idx], pred[idx], s=3, alpha=0.25, linewidths=0, color="#1769AA")
        ax.plot([lo, hi], [lo, hi], color="#B23", lw=1, ls="--", label="perfect prediction")
        ax.set_xlim(lo, hi * 1.02)
        ax.set_ylim(min(0.0, float(pred.min())), hi * 1.02)
        ax.set_xlabel("Actual demand (kWh)")
        ax.set_ylabel("Predicted demand (kWh)")
        ax.set_title(f"{label}\nR² = {r2:.4f} (all {len(df):,} rows)")
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(
        "Actual vs predicted hourly EV charging demand — chronological test set",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / "fig_actual_vs_predicted.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 2. Error distribution
# --------------------------------------------------------------------------


def plot_error_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    models = available_models(df)
    if not models:
        raise ValueError("no recognised prediction columns in the predictions file")
    actual = df[TARGET].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for col, label in models:
        resid = actual - df[col].to_numpy(dtype=float)
        axes[0].hist(resid, bins=200, range=(-400, 400), histtype="step", lw=1.3, label=label)
        axes[1].hist(
            np.abs(resid), bins=200, range=(0, 400), histtype="step", lw=1.3, label=label
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Residual: actual − predicted (kWh)")
    axes[0].set_ylabel("Number of test rows (log scale)")
    axes[0].set_title("Signed prediction error")
    axes[1].set_xlabel("Absolute error (kWh)")
    axes[1].set_ylabel("Number of test rows (log scale)")
    axes[1].set_title("Absolute error")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Prediction error distribution on the test set", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig_error_distribution.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 3. Model comparison
# --------------------------------------------------------------------------


def comparison_table(
    comparison_path: Path = DEFAULT_COMPARISON,
    baseline_metrics_path: Path = DEFAULT_BASELINE_METRICS,
) -> pd.DataFrame:
    """Model comparison rows including the persistence baseline (test split)."""
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"comparison table not found: {comparison_path}. "
            "Run `python -m src.models.train_models`."
        )
    rows = []
    if baseline_metrics_path.exists():
        import json

        with baseline_metrics_path.open(encoding="utf-8") as fh:
            b = json.load(fh)["test"]
        rows.append(
            {"model": "Persistence (lag_1)", "mae": b["mae"], "rmse": b["rmse"], "r2": b["r2"]}
        )
    comp = pd.read_csv(comparison_path)
    if "split" in comp.columns:
        comp = comp[comp["split"].astype(str).str.lower() == "test"]
    for _, r in comp.iterrows():
        rows.append({"model": r["model"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]})
    return pd.DataFrame(rows)


def plot_model_comparison(comparison: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    specs = [
        ("mae", "MAE (kWh)", "lower is better"),
        ("rmse", "RMSE (kWh)", "lower is better"),
        ("r2", "R²", "higher is better"),
    ]
    colors = ["#8C9BAB", "#4C90C8", "#1769AA"]
    for ax, (col, title, note) in zip(axes, specs):
        bars = ax.bar(comparison["model"], comparison[col], color=colors[: len(comparison)])
        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        if col == "r2":
            ax.set_ylim(min(0.0, float(comparison[col].min()) - 0.02), 1.0)
        for bar, value in zip(bars, comparison[col]):
            ax.annotate(
                f"{value:.4f}" if col == "r2" else f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Test-set performance — identical chronological test rows", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig_model_comparison.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 4. Per-zone error
# --------------------------------------------------------------------------


def zone_errors(df: pd.DataFrame, col: str) -> pd.DataFrame:
    err = (df[TARGET] - df[col]).abs()
    out = (
        pd.DataFrame({ZONE: df[ZONE], "abs_err": err, "actual": df[TARGET]})
        .groupby(ZONE)
        .agg(mae=("abs_err", "mean"), mean_actual=("actual", "mean"), n=("abs_err", "size"))
        .reset_index()
        .sort_values("mae", ascending=False)
    )
    return out


def plot_zone_error(df: pd.DataFrame, out_dir: Path, model_col: str, label: str, top: int = 20) -> Path:
    z = zone_errors(df, model_col)
    head = z.head(top).iloc[::-1]  # worst at the top of the horizontal chart

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(head[ZONE].astype(str), head["mae"], color="#1769AA")
    ax.set_xlabel("Mean absolute error (kWh)")
    ax.set_ylabel("Zone ID (TAZID)")
    ax.set_title(
        f"Worst {len(head)} of {len(z)} zones by MAE — {label}\n"
        f"dataset-wide MAE = {z['mae'].mean():.2f} kWh",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path = out_dir / "fig_zone_error.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 5. Error by demand level
# --------------------------------------------------------------------------


def plot_high_demand_errors(df: pd.DataFrame, out_dir: Path, bins: int = 10) -> Path:
    """Absolute-error profile across actual-demand deciles (all models).

    Answers: where does the error live — quiet hours or peak demand?
    """
    models = available_models(df)
    if not models:
        raise ValueError("no recognised prediction columns in the predictions file")

    data = df[[TARGET]].copy()
    data["decile"] = pd.qcut(
        data[TARGET].rank(method="first"), bins, labels=[f"{i + 1}" for i in range(bins)]
    )
    fig, ax = plt.subplots(figsize=(9, 4.3))
    width = 0.8 / len(models)
    x = np.arange(bins)
    for i, (col, label) in enumerate(models):
        abs_err = (data[TARGET] - df.loc[data.index, col]).abs()
        means = abs_err.groupby(data["decile"], observed=True).mean().reindex([str(b + 1) for b in range(bins)])
        ax.bar(x + (i - (len(models) - 1) / 2) * width, means.to_numpy(), width, label=label)
    decile_means = data.groupby("decile", observed=True)[TARGET].mean().reindex(
        [str(b + 1) for b in range(bins)]
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i + 1}\n~{v:.0f} kWh" for i, v in enumerate(decile_means)])
    ax.set_xlabel("Actual-demand decile (mean actual demand per decile)")
    ax.set_ylabel("Mean absolute error (kWh)")
    ax.set_title("Mean absolute error by actual-demand level — test set", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out_dir / "fig_high_demand_errors.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def generate_all(
    predictions_path: Path = DEFAULT_PREDICTIONS,
    comparison_path: Path = DEFAULT_COMPARISON,
    baseline_metrics_path: Path = DEFAULT_BASELINE_METRICS,
    out_dir: Path = DEFAULT_FIGURES_DIR,
    model_for_zone_plot: str = "random_forest_predicted_kwh",
) -> dict:
    """Generate every figure; returns ``{name: path}`` plus a facts summary."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(predictions_path)
    comparison = comparison_table(comparison_path, baseline_metrics_path)

    n_missing_actual = int(df[TARGET].isna().sum())
    if n_missing_actual:
        raise AssertionError(f"{n_missing_actual} test rows have a missing actual value")

    models = available_models(df)
    zone_col = model_for_zone_plot if any(c == model_for_zone_plot for c, _ in models) else models[-1][0]
    zone_label = dict(models)[zone_col]

    written = {
        "actual_vs_predicted": plot_actual_vs_predicted(df, out_dir),
        "error_distribution": plot_error_distribution(df, out_dir),
        "model_comparison": plot_model_comparison(comparison, out_dir),
        "zone_error": plot_zone_error(df, out_dir, zone_col, zone_label),
        "high_demand_errors": plot_high_demand_errors(df, out_dir),
    }

    z = zone_errors(df, zone_col)
    facts = {
        "test_rows": int(len(df)),
        "zones": int(df[ZONE].nunique()),
        "models": {label: col for col, label in models},
        "comparison": comparison.to_dict(orient="records"),
        "zone_mae_median": float(z["mae"].median()),
        "zone_mae_worst": {"zone": str(z.iloc[0][ZONE]), "mae": float(z.iloc[0]["mae"])},
        "zone_mae_best": {"zone": str(z.iloc[-1][ZONE]), "mae": float(z.iloc[-1]["mae"])},
        "zone_plot_model": zone_label,
        "figures": {k: str(v) for k, v in written.items()},
    }
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation figures.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args(argv)

    try:
        facts = generate_all(
            args.predictions, args.comparison, args.baseline_metrics, args.out_dir
        )
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    import json

    print(json.dumps(facts, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
