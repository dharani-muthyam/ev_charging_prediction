"""Plain-language helpers for the dashboard's prediction and evaluation sections.

The dashboard is a read-only view over results the offline pipeline wrote. The
functions here only *describe* those numbers — they never recompute, refit or
alter a metric. Keeping them in a plain module (rather than inside the
Streamlit script) lets them be unit-tested without launching an app.

Nothing here invents results: every value passed in must come from the saved
artefacts (``reports/baseline_metrics.json``, ``results/tables/model_comparison.csv``).
"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

DAY_NAMES: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)
MONTH_NAMES: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# The reference every trained model must beat, as named in the comparison table.
BASELINE_LABEL = "Persistence baseline (lag_1)"

# One-line, beginner-friendly meaning of each metric. Kept short so it reads on
# a phone; the longer form lives in the dashboard expander.
METRIC_SHORT: dict[str, str] = {
    "MAE (kWh)": "How far off a prediction is, on average, in kWh.",
    "RMSE (kWh)": "Like MAE, but big misses count more. Usually higher than MAE.",
    "R²": "Share of the variation the model explains: 1.0 is perfect, 0.0 is no better than guessing the average.",
}


def format_kwh(value: float, decimals: int = 2) -> str:
    """Format an energy value with an explicit kWh unit."""
    return f"{value:,.{decimals}f} kWh"


def describe_metric(key: str) -> str:
    """Short plain-English definition for a comparison-table column."""
    return METRIC_SHORT.get(key, "")


def _num(value) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if it is not usable."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


# Human-readable names for the two demand inputs that must be non-negative.
DEMAND_INPUT_LABELS: dict[str, str] = {
    "lag_1": "Last hour's demand",
    "lag_24": "Demand at this hour yesterday",
}


def validate_demand_inputs(values: Mapping[str, object]) -> list[str]:
    """Return beginner-friendly error messages for unusable demand inputs.

    Energy delivered in an hour cannot be negative, so a negative ``lag_1`` or
    ``lag_24`` is rejected here *before* any model is called. Non-numeric and
    non-finite values are rejected for the same reason.

    An empty list means every demand input is usable. Zero is valid — a zone
    with no charging in the previous hour is a real, common situation.
    """
    problems: list[str] = []
    for key, label in DEMAND_INPUT_LABELS.items():
        if key not in values:
            continue
        number = _num(values[key])
        # ``_num`` accepts any finite float; ``inf`` is not usable as a demand.
        if number is None or not math.isfinite(number):
            problems.append(
                f"**{label}** must be a number — please enter a value of 0 or more kWh."
            )
        elif number < 0:
            problems.append(
                f"**{label}** cannot be negative. Charging demand is the energy a zone "
                f"delivered in an hour, so it must be **0 kWh or more** "
                f"(you entered {number:,.2f} kWh)."
            )
    return problems


def column_context(comparison: pd.DataFrame, model_name: str) -> dict | None:
    """Test-set MAE/R² for ``model_name`` from the comparison table.

    Accepts either the dashboard's table (a ``Model`` column) or a lookup frame
    indexed by model name, so callers do not have to reshape it first. Returns
    ``None`` when the table is empty or the model has no usable row, so the
    caller can simply skip the explanatory sentence.
    """
    if comparison is None or getattr(comparison, "empty", True):
        return None

    if "Model" in comparison.columns:
        rows = comparison.loc[comparison["Model"] == model_name]
    elif model_name in comparison.index:
        # ``.loc`` on a duplicated index would return a frame, not a row.
        rows = comparison.loc[[model_name]] if comparison.index.is_unique else pd.DataFrame()
    else:
        return None

    if rows.empty:
        return None
    row = rows.iloc[0]
    mae = _num(row.get("MAE (kWh)"))
    r2 = _num(row.get("R²"))
    if mae is None:
        return None
    return {"mae": mae, "r2": r2}


def improvement_vs_baseline(comparison: pd.DataFrame, model_name: str) -> dict | None:
    """Percentage change in MAE against the persistence baseline.

    A negative number means the model's average error is lower than simply
    repeating last hour's demand. Returns ``None`` when either value is missing
    or the baseline MAE is zero, rather than producing a meaningless figure.
    """
    base = column_context(comparison, BASELINE_LABEL)
    model = column_context(comparison, model_name)
    if base is None or model is None or base["mae"] == 0:
        return None
    return {
        "mae_delta": model["mae"] - base["mae"],
        "mae_pct": (model["mae"] - base["mae"]) / base["mae"] * 100.0,
        "baseline_mae": base["mae"],
        "model_mae": model["mae"],
    }


def comparison_summary(comparison: pd.DataFrame) -> dict:
    """Headline facts about the comparison table, computed from its own rows.

    Returns best-model names derived from the table, never hard-coded, plus the
    baseline row when it is present. An empty table yields empty ``best``/
    ``baseline`` entries so the caller can show a neutral message.
    """
    summary: dict = {"best": {}, "baseline": None, "n_models": 0}
    if comparison is None or comparison.empty:
        return summary

    summary["n_models"] = int(len(comparison))
    for key, column, mode in (
        ("mae", "MAE (kWh)", "min"),
        ("rmse", "RMSE (kWh)", "min"),
        ("r2", "R²", "max"),
    ):
        series = pd.to_numeric(comparison.get(column), errors="coerce")
        if series.notna().any():
            idx = series.idxmin() if mode == "min" else series.idxmax()
            summary["best"][key] = {
                "model": str(comparison.loc[idx, "Model"]),
                "value": float(series.loc[idx]),
            }

    summary["baseline"] = column_context(comparison, BASELINE_LABEL)
    if summary["baseline"] is not None:
        summary["baseline"]["model"] = BASELINE_LABEL
    return summary


def explain_comparison(
    comparison: pd.DataFrame, best_mae: Mapping | None = None, best_r2: Mapping | None = None
) -> list[str]:
    """Build short factual sentences describing the comparison table.

    Every sentence is derived from the rows passed in. Claims that cannot be
    supported by the data (for example, "better on all three metrics") are only
    made when the numbers actually show it.
    """
    summary = comparison_summary(comparison)
    if summary["n_models"] == 0:
        return []

    lines: list[str] = []
    best_mae = best_mae or summary["best"].get("mae")
    best_r2 = best_r2 or summary["best"].get("r2")
    base = summary["baseline"]
    winners = {k: v["model"] for k, v in summary["best"].items() if v}
    single_winner = len(winners) == 3 and len(set(winners.values())) == 1

    if best_mae:
        line = (
            f"**{best_mae['model']}** has the lowest average error "
            f"(MAE {format_kwh(best_mae['value'])})."
        )
        if base and base["mae"]:
            delta = (best_mae["value"] - base["mae"]) / base["mae"] * 100.0
            if delta < 0:
                line += (
                    f" That is {abs(delta):.0f}% lower than repeating the previous "
                    f"hour's demand ({format_kwh(base['mae'])})."
                )
        lines.append(line)

    if best_r2:
        if single_winner and best_r2["model"] == best_mae["model"]:
            lines.append(
                f"It also explains the most variation in demand "
                f"(R² {best_r2['value']:.4f}) and is best on all three metrics, "
                "so it is the strongest of the evaluated approaches."
            )
        else:
            lines.append(
                f"**{best_r2['model']}** explains the most variation in demand "
                f"(R² {best_r2['value']:.4f})."
            )
            # Only claim an overall win when one model genuinely holds all three.
            if single_winner:
                lines.append(
                    f"**{next(iter(winners.values()))}** is best on all three metrics, "
                    "so it is the strongest of the evaluated approaches."
                )
            elif winners:
                lines.append(
                    "No single model wins on every metric, so each row is worth "
                    "reading individually."
                )

    lines.append(
        "Lower MAE and RMSE are better; higher R² is better. All rows use the same "
        "chronological test rows, so they are directly comparable."
    )
    return lines
