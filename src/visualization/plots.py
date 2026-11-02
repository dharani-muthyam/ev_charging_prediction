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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results" / "predictions" / "model_test_predictions.csv"
DEFAULT_COMPARISON = PROJECT_ROOT / "results" / "tables" / "model_comparison.csv"
DEFAULT_BASELINE_METRICS = PROJECT_ROOT / "reports" / "baseline_metrics.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
# Full (unmodified) long-form demand table, used only for demand-profile charts.
DEFAULT_DEMAND_LONG = PROJECT_ROOT / "data" / "processed" / "demand_long.csv"

TARGET = "demand_kwh"
ZONE = "zone_id"

# Display order and colours for the three evaluated approaches.
MODEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("baseline_predicted_kwh", "Persistence (lag_1)"),
    ("linear_regression_predicted_kwh", "Linear Regression"),
    ("random_forest_predicted_kwh", "Random Forest"),
)

DPI = 120

# One palette for every figure so the report and the dashboard agree.
INK = "#163A63"        # titles / axis text
MUTED = "#526579"      # tick labels, captions
ACCENT = "#1769AA"     # primary series
GUIDE = "#B23"         # reference lines (perfect prediction, zero error)
GRID = "#D8E5F2"
SERIES_COLORS: tuple[str, ...] = ("#8C9BAB", "#4C90C8", "#1769AA")
# Wider-separated palette for overlapping line charts, where similar blues merge.
LINE_COLORS: tuple[str, ...] = ("#8C9BAB", "#D9882B", "#1769AA")

# Readable label for each prediction column, shared by charts and captions.
MODEL_LABELS: dict[str, str] = dict(MODEL_COLUMNS)


class _LegendHandler(HandlerBase):
    """Legend key for step histograms (a bare Patch would draw a filled box)."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        line = Line2D([xdescent, xdescent + width], [0, 0], lw=1.6, color=orig_handle[0])
        line.set_transform(trans)
        return [line]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.9,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 10,
        }
    )


def _label_axes(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Set axis labels with units and keep tick labels from touching."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def _thin_ticks(ax, axis: str = "x", max_ticks: int = 8) -> None:
    """Show at most ``max_ticks`` labels on ``axis`` so long lists stay readable."""
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    if len(ticks) > max_ticks:
        step = int(np.ceil(len(ticks) / max_ticks))
        keep = ticks[::step]
        if axis == "x":
            ax.set_xticks(keep)
        else:
            ax.set_yticks(keep)


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
    """Predicted vs actual demand, one panel per model, with MAE and R².

    The dashed line is perfect prediction; a cloud hugging it is a good fit.
    Both axes share the same kWh scale so panels are directly comparable.
    """
    models = available_models(df)
    if not models:
        raise ValueError("no recognised prediction columns in the predictions file")

    actual = df[TARGET].to_numpy(dtype=float)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.6), squeeze=False)
    # Subsample for legibility only; all points are used for the reported R²/MAE.
    limit = 8000
    idx = (
        np.random.default_rng(42).choice(len(df), size=limit, replace=False)
        if len(df) > limit
        else np.arange(len(df))
    )
    lo, hi = float(np.min(actual)), float(np.max(actual))
    pad = (hi - lo) * 0.02

    for ax, (col, label) in zip(axes[0], models):
        pred = df[col].to_numpy(dtype=float)
        resid = actual - pred
        rr = 1.0 - float(np.sum(resid**2)) / float(np.sum((actual - actual.mean()) ** 2))
        mae = float(np.mean(np.abs(resid)))
        ax.scatter(
            actual[idx], pred[idx], s=3, alpha=0.22, linewidths=0,
            color=ACCENT, rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color=GUIDE, lw=1.1, ls="--", label="perfect prediction")
        ax.set_xlim(lo, hi + pad)
        ax.set_ylim(min(0.0, float(pred.min())), hi + pad)
        ax.set_aspect("equal", adjustable="box")
        _label_axes(ax, "Actual demand (kWh)", "Predicted demand (kWh)")
        ax.set_title(f"{label}", loc="left")
        ax.text(
            0.97, 0.06,
            f"MAE {mae:,.2f} kWh\nR² {rr:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=INK, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, alpha=0.9),
        )
    axes[0][0].legend(loc="upper left")

    fig.suptitle(
        "Actual vs predicted hourly demand — chronological test set "
        f"({len(df):,} rows, {df[ZONE].nunique()} zones)",
        fontsize=12, color=INK,
    )
    fig.text(
        0.5, -0.02,
        f"Each panel shows a random {len(idx):,}-row sample for legibility; "
        "MAE and R² use every test row.",
        ha="center", fontsize=8, color=MUTED,
    )
    fig.tight_layout()
    path = out_dir / "fig_actual_vs_predicted.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 2. Error distribution
# --------------------------------------------------------------------------


def plot_error_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Signed and absolute prediction errors for every model.

    A log count axis is used because most rows have small errors and a few have
    very large ones; on a linear axis the large errors would be invisible.
    """
    models = available_models(df)
    if not models:
        raise ValueError("no recognised prediction columns in the predictions file")
    actual = df[TARGET].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for i, (col, label) in enumerate(models):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        resid = actual - df[col].to_numpy(dtype=float)
        axes[0].hist(
            resid, bins=160, range=(-300, 300), histtype="step", lw=1.4,
            color=color, label=label,
        )
        axes[1].hist(
            np.abs(resid), bins=160, range=(0, 300), histtype="step", lw=1.4,
            color=color, label=label,
        )
        mae = float(np.mean(np.abs(resid)))
        axes[1].axvline(
            mae, color=color, lw=1, ls=":", alpha=0.9,
        )
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Number of test rows (log scale)")
    axes[0].axvline(0, color=GUIDE, lw=1, ls="--")
    axes[0].text(0.02, 0.95, "zero error", transform=axes[0].transAxes,
                 fontsize=8, color=GUIDE, va="top")
    _label_axes(axes[0], "Signed error: actual − predicted (kWh)")
    _label_axes(axes[1], "Absolute error (kWh)")
    axes[0].set_title("Signed error — is the model biased?", loc="left")
    axes[1].set_title("Absolute error — dotted line = MAE", loc="left")

    handles = [Line2D([0], [0], color=c, lw=1.6) for c in LINE_COLORS[: len(models)]]
    axes[1].legend(handles, [lab for _, lab in models], loc="upper right")

    fig.suptitle("Prediction error distribution — test set", fontsize=12, color=INK)
    fig.text(
        0.5, -0.02,
        "Signed error centred on zero means little systematic bias; the long right "
        "tail in absolute error is a few very high-demand hours.",
        ha="center", fontsize=8, color=MUTED,
    )
    fig.tight_layout()
    path = out_dir / "fig_error_distribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
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
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    specs = [
        ("mae", "MAE (kWh)", "lower is better"),
        ("rmse", "RMSE (kWh)", "lower is better"),
        ("r2", "R²", "higher is better"),
    ]
    colors = SERIES_COLORS
    for ax, (col, title, note) in zip(axes, specs):
        bars = ax.bar(
            comparison["model"], comparison[col],
            color=colors[: len(comparison)], width=0.62, edgecolor="white",
        )
        ax.set_title(f"{title}  ·  {note}", loc="left")
        ax.set_xticks(range(len(comparison)))
        ax.set_xticklabels(
            [m.replace("Persistence (lag_1)", "Persistence\n(lag_1)") for m in comparison["model"]],
            fontsize=8,
        )
        ax.grid(axis="x", visible=False)
        top = float(comparison[col].max())
        ax.set_ylim(0, top * 1.18 if top > 0 else 1)
        for bar, value in zip(bars, comparison[col]):
            ax.annotate(
                f"{value:.4f}" if col == "r2" else f"{value:,.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5, color=INK,
            )
    _label_axes(axes[0], None, "kWh (lower is better)")
    _label_axes(axes[1], None, "kWh (lower is better)")
    _label_axes(axes[2], None, "Share of variance explained")
    fig.suptitle(
        "Test-set performance — identical chronological test rows",
        fontsize=12, color=INK,
    )
    fig.tight_layout()
    path = out_dir / "fig_model_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
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
    """Worst zones by MAE, drawn against the dataset-wide MAE for context."""
    z = zone_errors(df, model_col)
    head = z.head(top).iloc[::-1]  # worst at the top of the horizontal chart
    overall = float(z["mae"].mean())
    median = float(z["mae"].median())

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(head[ZONE].astype(str), head["mae"], color=ACCENT, height=0.72)
    ax.axvline(
        overall, color=GUIDE, lw=1.2, ls="--",
        label=f"mean MAE across zones ({overall:,.2f} kWh)",
    )
    for bar, value in zip(bars, head["mae"]):
        ax.annotate(
            f"{value:,.1f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=MUTED,
        )
    _label_axes(ax, "Mean absolute error (kWh)", "Zone ID (TAZID)")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"Worst {len(head)} of {len(z)} zones by MAE — {label}",
        loc="left",
    )
    ax.legend(loc="lower right")
    fig.text(
        0.01, -0.01,
        f"Median zone MAE {median:,.2f} kWh. High-error zones are the dense, "
        "high-demand zones, so per-zone forecasts there are less reliable.",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout()
    path = out_dir / "fig_zone_error.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 5. Error by demand level
# --------------------------------------------------------------------------


def plot_high_demand_errors(df: pd.DataFrame, out_dir: Path, bins: int = 10) -> Path:
    """Absolute-error profile across actual-demand deciles (all models).

    Answers: where does the error live — quiet hours or peak demand? A log y-axis
    is used because the top decile's error is an order of magnitude above the
    rest; on a linear axis the lower deciles would be invisible.
    """
    models = available_models(df)
    if not models:
        raise ValueError("no recognised prediction columns in the predictions file")

    data = df[[TARGET]].copy()
    data["decile"] = pd.qcut(
        data[TARGET].rank(method="first"), bins, labels=[f"{i + 1}" for i in range(bins)]
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.8 / len(models)
    x = np.arange(bins)
    for i, (col, label) in enumerate(models):
        abs_err = (data[TARGET] - df.loc[data.index, col]).abs()
        means = abs_err.groupby(data["decile"], observed=True).mean().reindex([str(b + 1) for b in range(bins)])
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width, means.to_numpy(), width,
            label=label, color=SERIES_COLORS[i % len(SERIES_COLORS)],
        )
    decile_means = data.groupby("decile", observed=True)[TARGET].mean().reindex(
        [str(b + 1) for b in range(bins)]
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i + 1}\n~{v:,.0f} kWh" for i, v in enumerate(decile_means)])
    ax.set_yscale("log")
    means = [
        (data[TARGET] - df.loc[data.index, c]).abs()
        .groupby(data["decile"], observed=True).mean()
        for c, _ in models
    ]
    lo = max(float(min(m.min() for m in means)), 1e-3) * 0.6
    hi = float(max(m.max() for m in means)) * 1.6
    ax.set_ylim(lo, hi)
    _label_axes(
        ax,
        "Actual-demand decile (mean actual demand in the decile)",
        "Mean absolute error (kWh, log scale)",
    )
    ax.grid(axis="x", visible=False)
    # Fixed, rounded tick labels keep the log axis readable.
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_title("Mean absolute error by actual-demand level — test set", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = out_dir / "fig_high_demand_errors.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 6. Demand profiles (hour of day / day of week)
# --------------------------------------------------------------------------


DAY_NAMES: tuple[str, ...] = (
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
)


def load_demand_long(path: Path = DEFAULT_DEMAND_LONG) -> pd.DataFrame:
    """Load the full long-form demand table (``time, zone_id, demand_kwh``).

    This is the unmodified preprocessing output; it is read-only here and is
    used only to describe the demand series, never to retrain anything.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"demand table not found: {path}. Run `python -m src.preprocessing.prepare_data`."
        )
    df = pd.read_csv(path, usecols=["time", ZONE, TARGET], parse_dates=["time"])
    if df.empty:
        raise ValueError(f"demand table is empty: {path}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    return df


def hourly_profile(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean demand by hour of day and the mean across zones at each timestamp."""
    per_hour = df[TARGET].groupby(df["time"].dt.hour).mean()
    per_hour.index.name = "hour"
    # Averaging zones per timestamp gives the typical zone-hour, not a sum over
    # 275 zones that would just measure how many zones exist.
    zone_mean = df.groupby("time")[TARGET].mean()
    by_hour = zone_mean.groupby(zone_mean.index.hour).mean()
    return per_hour, by_hour


def plot_hourly_demand_profile(
    df: pd.DataFrame,
    out_dir: Path,
    test_predictions: pd.DataFrame | None = None,
) -> Path:
    """Mean demand by hour of day for the full period.

    When test predictions are supplied, the test-window actual demand is
    overlaid so the reader can see how the forecast period compares with the
    whole six months. Both lines are mean kWh per zone-hour.
    """
    per_hour, _ = hourly_profile(df)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(
        per_hour.index, per_hour.to_numpy(), color=ACCENT, lw=2.2,
        marker="o", ms=4, label=f"full period ({len(df):,} zone-hours)",
    )

    if test_predictions is not None and not test_predictions.empty:
        test = test_predictions[["time", TARGET]].dropna()
        if not test.empty:
            test_hour = test[TARGET].groupby(test["time"].dt.hour).mean()
            ax.plot(
                test_hour.index, test_hour.to_numpy(), color=GUIDE, lw=1.6,
                ls="--", marker="s", ms=3,
                label=f"test window ({len(test):,} rows)",
            )

    peak_hour = int(per_hour.idxmax())
    trough_hour = int(per_hour.idxmin())
    ax.axvline(peak_hour, color=MUTED, lw=0.9, ls=":")
    # Put the callout below the series and away from the legend, which sits in
    # the top-left corner; xytext in points keeps it off the plotted line.
    ax.annotate(
        f"peak {peak_hour:02d}:00",
        (peak_hour, float(per_hour.min())),
        xytext=(8, 6), textcoords="offset points",
        fontsize=8.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, alpha=0.9),
    )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    _label_axes(ax, "Hour of day (local time, 24-hour)", "Mean demand (kWh per zone-hour)")
    ax.grid(axis="x", visible=False)
    ax.set_title(
        f"Hourly demand profile — mean across {df[ZONE].nunique()} zones",
        loc="left",
    )
    ax.set_ylim(0, float(per_hour.max()) * 1.12)
    ax.legend(loc="lower center", ncol=2)
    ax.margins(x=0.02)
    fig.text(
        0.01, -0.01,
        f"Trough {trough_hour:02d}:00 ↔ peak {peak_hour:02d}:00. "
        "Demand is averaged per zone-hour, so the y-axis is a typical zone, not a city total.",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout()
    path = out_dir / "fig_hourly_demand_profile.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_daily_demand_profile(df: pd.DataFrame, out_dir: Path) -> Path:
    """Mean demand by day of week, split into weekday hours vs weekend hours.

    Two panels share one y-axis so weekday/weekend levels are comparable.
    """
    frame = pd.DataFrame(
        {
            "day": df["time"].dt.dayofweek,
            "hour": df["time"].dt.hour,
            "kwh": df[TARGET].to_numpy(),
        }
    )
    per_day = frame.groupby("day")["kwh"].mean()
    weekend = frame["day"] >= 5

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    ax = axes[0]
    bars = ax.bar(
        [DAY_NAMES[d] for d in per_day.index], per_day.to_numpy(),
        color=[ACCENT if d < 5 else "#4C90C8" for d in per_day.index], width=0.66,
    )
    for bar, value in zip(bars, per_day.to_numpy()):
        ax.annotate(
            f"{value:,.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8, color=INK,
        )
    _label_axes(ax, "Day of week", "Mean demand (kWh per zone-hour)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Mean demand by day of week", loc="left")

    ax = axes[1]
    for label, mask, color in (
        ("Weekdays (Mon–Fri)", ~weekend, ACCENT),
        ("Weekend (Sat–Sun)", weekend, "#D9882B"),
    ):
        prof = frame.loc[mask, "kwh"].groupby(frame.loc[mask, "hour"]).mean()
        ax.plot(prof.index, prof.to_numpy(), color=color, lw=2, marker="o", ms=3, label=label)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    _label_axes(ax, "Hour of day (24-hour)")
    ax.grid(axis="x", visible=False)
    # The shared y-axis lives on the left panel only; do not draw it twice.
    ax.tick_params(axis="y", length=0)
    ax.set_title("Hourly shape: weekdays vs weekend", loc="left")
    ax.set_ylim(0, float(frame.groupby(["hour", "day"])["kwh"].mean().max()) * 1.15)
    ax.legend(loc="upper right")
    ax.margins(x=0.02)

    fig.suptitle(
        f"Daily demand patterns — all {len(df):,} zone-hours, "
        f"{df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d}",
        fontsize=12, color=INK,
    )
    fig.tight_layout()
    path = out_dir / "fig_daily_demand_profile.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 7. Driver
# --------------------------------------------------------------------------


def generate_all(
    predictions_path: Path = DEFAULT_PREDICTIONS,
    comparison_path: Path = DEFAULT_COMPARISON,
    baseline_metrics_path: Path = DEFAULT_BASELINE_METRICS,
    out_dir: Path = DEFAULT_FIGURES_DIR,
    model_for_zone_plot: str = "random_forest_predicted_kwh",
    demand_path: Path = DEFAULT_DEMAND_LONG,
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

    # Demand-profile figures describe the series itself, so they need the long
    # table. If it is absent the other figures are still valid and are kept.
    profile_facts: dict = {}
    if demand_path.exists():
        demand = load_demand_long(demand_path)
        per_hour, _ = hourly_profile(demand)
        written["hourly_demand_profile"] = plot_hourly_demand_profile(
            demand, out_dir, test_predictions=df
        )
        written["daily_demand_profile"] = plot_daily_demand_profile(demand, out_dir)
        day_means = demand[TARGET].groupby(demand["time"].dt.dayofweek).mean()
        profile_facts = {
            "demand_rows": int(len(demand)),
            "demand_window": [str(demand["time"].min()), str(demand["time"].max())],
            "demand_peak_hour": int(per_hour.idxmax()),
            "demand_trough_hour": int(per_hour.idxmin()),
            "demand_busiest_day": DAY_NAMES[int(day_means.idxmax())],
            "demand_quietest_day": DAY_NAMES[int(day_means.idxmin())],
        }
    else:
        profile_facts = {"demand_profile_figures": "skipped (demand_long.csv not found)"}

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
        **profile_facts,
    }
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation figures.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--demand", type=Path, default=DEFAULT_DEMAND_LONG)
    args = parser.parse_args(argv)

    try:
        facts = generate_all(
            args.predictions, args.comparison, args.baseline_metrics, args.out_dir,
            demand_path=args.demand,
        )
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    import json

    print(json.dumps(facts, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
