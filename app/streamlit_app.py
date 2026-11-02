"""EV charging demand dashboard (Streamlit).

Read-only UI over the artefacts the offline pipeline produced:

    models/linear_regression.joblib, models/random_forest.joblib
    results/tables/model_comparison.csv          (test-set metrics)
    reports/baseline_metrics.json                (persistence baseline)
    results/predictions/model_test_predictions.csv
    reports/experiments/validation_study.json    (optional, if the study was run)
    results/figures/*.png                        (optional)

Nothing is retrained here and no missing value is ever filled in: if an artefact
is absent the section says so and is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app.dashboard_text import (
    DAY_NAMES,
    MONTH_NAMES,
    column_context,
    describe_metric,
    explain_comparison,
    format_kwh,
    improvement_vs_baseline,
    validate_demand_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = ROOT / "results" / "tables"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
FIGURES_DIR = ROOT / "results" / "figures"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"

TARGET = "demand_kwh"
ZONE = "zone_id"

st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide",
)

# -------------------- Blue & white theme (single consolidated block) ---------
st.markdown(
    """
    <style>
    .stApp { background-color: #F5F8FC; color: #1E293B; }
    /* The Streamlit header is position:absolute over the top of the scroll
       area, so with the default 2rem padding the first heading sits under it
       and is clipped on load. Clear the header height explicitly. */
    header[data-testid="stHeader"] { background: transparent; }
    .block-container {
        padding-top: calc(var(--st-header-height, 3.75rem) + 1rem) !important;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    h1, h2, h3 { color: #163A63 !important; font-weight: 700 !important; }
    a { color: #1769AA !important; }

    /* Section headings get a subtle rule so long pages stay scannable. */
    h2 {
        border-bottom: 1px solid #D8E5F2;
        padding-bottom: 0.35rem;
        margin-top: 0.5rem !important;
    }

    /* Streamlit 1.63's default accent is red (#FF4B4B), and `--primary-color`
       does not reach the slider. The styled components in Slider.js give their
       names away through the emotion target ids, so match those rather than
       the DOM shape: efbyxod3 = 12px round thumb, efbyxod5 = 4px track bar
       (grey rail + filled part), efbyxod4 = the floating value label. */
    [class*="efbyxod3"] { background-color: #1769AA !important; }
    /* Floating slider value label (bold blue, no chip background). */
    [data-testid="stSliderThumbValue"],
    [data-testid="stSliderThumbValue"] * {
        color: #1769AA !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }
    /* Track: keep the grey rail, paint the filled part with a hard stop so the
       gradient does not blend. The stop is a fixed position (not a live value),
       which is why the fraction below is left explicit. */
    [class*="efbyxod5"] {
        background: linear-gradient(
            to right,
            #1769AA 0%,
            #1769AA var(--ev-slider-fill, 36%),
            #D8E5F2 var(--ev-slider-fill, 36%),
            #D8E5F2 100%
        ) !important;
    }

    .stButton > button, .stDownloadButton > button {
        background-color: #1769AA !important;
        color: white !important;
        border: 1px solid #1769AA !important;
        border-radius: 8px !important;
        padding: 0.55rem 1rem !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #104E82 !important;
        border-color: #104E82 !important;
        color: white !important;
    }

    /* Inputs and selects: white cards on the soft page background. */
    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div {
        border-radius: 8px !important;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #D8E5F2;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(22, 58, 99, 0.06);
    }
    [data-testid="stMetricLabel"] { color: #526579 !important; }
    [data-testid="stMetricValue"] { color: #1769AA !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-weight: 600 !important; }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 14px rgba(22, 58, 99, 0.12);
        border-color: #B9CDE4;
    }

    /* Tabs, tables and expanders: same card language as the metric cards. */
    [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
        border: 1px solid #D8E5F2;
        border-radius: 10px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid #D8E5F2; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #526579;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] { color: #1769AA !important; }
    [data-testid="stExpander"] details {
        border: 1px solid #D8E5F2;
        border-radius: 10px;
        background-color: #FFFFFF;
    }

    /* Small tags for shortened wording on narrow screens. */
    .ev-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.25rem 0 0.5rem 0; }
    .ev-chip {
        background-color: #E8F0FA;
        color: #163A63;
        border: 1px solid #D8E5F2;
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Narrow screens: breathe out the padding and drop the title size a notch. */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        h1 { font-size: 1.6rem !important; }
        h2 { font-size: 1.25rem !important; }
        [data-testid="stMetric"] { padding: 12px; }
        [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    }

    hr { border-color: #D8E5F2 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ EV Charging Demand Prediction")
st.caption("Machine learning dashboard | One-hour-ahead demand forecasting")
st.markdown(
    "Predict the charging volume (kWh) a zone will consume in the next hour, "
    "compare the trained models against the persistence baseline, and inspect "
    "the saved test-set results."
)


# -------------------- Cached loaders (no retraining, no per-click recompute) --
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str):
    """Load a saved estimator once per process. Raises on corruption."""
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_csv(path: str, parse_time: bool = False) -> pd.DataFrame | None:
    if not Path(path).is_file():
        return None
    try:
        return pd.read_csv(
            path,
            float_precision="round_trip",
            parse_dates=["time"] if parse_time else None,
        )
    except Exception:  # noqa: BLE001 - surfaced to the user by the caller
        return None


@st.cache_data(show_spinner=False)
def error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Absolute per-row errors for every model, keyed by its display label.

    ``predictions`` is the cached frame, so this aggregate is computed once per
    session rather than on every widget interaction.
    """
    err = predictions[[ZONE, TARGET]].copy()
    for label, col in (
        ("Persistence baseline (lag_1)", "baseline_predicted_kwh"),
        ("Linear Regression", "linear_regression_predicted_kwh"),
        ("Random Forest", "random_forest_predicted_kwh"),
    ):
        if col in predictions.columns:
            pred = pd.to_numeric(predictions[col], errors="coerce")
            err[label] = (pred - err[TARGET]).abs()
    return err


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | None:
    if not Path(path).is_file():
        return None
    try:
        with Path(path).open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


model_files = {
    "Linear Regression": MODELS_DIR / "linear_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
}
available_models = {name: path for name, path in model_files.items() if path.is_file()}
missing_models = [name for name in model_files if name not in available_models]

# Short descriptive tags under the header (pure presentation, nothing computed).
chips = [
    "One-hour-ahead horizon",
    "Hourly zone-level demand",
    "Persistence baseline + trained regressors",
    "Read-only over saved artefacts",
]
st.markdown(
    '<div class="ev-chips">'
    + "".join(f'<span class="ev-chip">{c}</span>' for c in chips)
    + "</div>",
    unsafe_allow_html=True,
)


# -------------------- Verified reference metrics -----------------------------
def build_comparison() -> pd.DataFrame:
    """Baseline + trained models on the same test rows, from saved artefacts."""
    rows: list[dict] = []

    baseline = load_json(str(REPORTS_DIR / "baseline_metrics.json"))
    if baseline and "test" in baseline:
        t = baseline["test"]
        rows.append(
            {
                "Model": "Persistence baseline (lag_1)",
                "MAE (kWh)": t.get("mae"),
                "RMSE (kWh)": t.get("rmse"),
                "R²": t.get("r2"),
                "Test rows": t.get("rows"),
            }
        )

    comparison = load_csv(str(TABLES_DIR / "model_comparison.csv"))
    if comparison is not None:
        comparison.columns = [str(c).strip() for c in comparison.columns]
        if "split" in comparison.columns:
            comparison = comparison[comparison["split"].astype(str).str.lower() == "test"]
        for _, r in comparison.iterrows():
            rows.append(
                {
                    "Model": r.get("model"),
                    "MAE (kWh)": pd.to_numeric(r.get("mae"), errors="coerce"),
                    "RMSE (kWh)": pd.to_numeric(r.get("rmse"), errors="coerce"),
                    "R²": pd.to_numeric(r.get("r2"), errors="coerce"),
                    "Test rows": None,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ("MAE (kWh)", "RMSE (kWh)", "R²"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


comparison_df = build_comparison()

# =====================================================================
# 1. Prediction
# =====================================================================
st.divider()
st.header("1. Predict charging demand")
st.caption(
    "Describe one zone at one moment in time. The selected model estimates how many "
    "kilowatt-hours (kWh) that zone will deliver in the **next hour**. "
    "This is a single what-if estimate — not the test-set results, which are in "
    "section 2."
)

if not available_models:
    st.error(
        "No trained models were found in `models/`, so nothing can be predicted yet. "
        "Generate them with `python -m src.models.train_models` (see the README), "
        "then reload this page."
    )
    st.caption(
        "The rest of the dashboard — model comparison, test-set results and figures "
        "— reads saved reports and still works without the model files, if those "
        "reports exist."
    )
else:
    if missing_models:
        st.warning(
            "Only some model files were found. Missing: "
            + ", ".join(f"`{name}`" for name in missing_models)
            + ". The models that are present can still be used."
        )

    # Show each model's verified test performance next to its selector entry.
    metric_lookup = (
        comparison_df.set_index("Model") if not comparison_df.empty else pd.DataFrame()
    )

    def _label(name: str) -> str:
        if metric_lookup.empty or name not in metric_lookup.index:
            return name
        row = metric_lookup.loc[name]
        return f"{name} — test MAE {row['MAE (kWh)']:.2f} kWh, R² {row['R²']:.4f}"

    form_col, result_col = st.columns([1.15, 1], gap="large")

    with form_col:
        with st.container(border=True):
            st.markdown("##### Forecast inputs")
            st.caption(
                "Not sure what to enter? The default values are a realistic evening "
                "example — press the button to see how a prediction looks."
            )
            model_name = st.selectbox(
                "Choose a trained model",
                options=list(available_models.keys()),
                format_func=_label,
                help="Both models use the same inputs. The label shows each model's "
                "test-set error so you can compare them before choosing.",
            )

            st.markdown("**Recent charging activity**")
            left, right = st.columns(2)
            with left:
                lag_1 = st.number_input(
                    "Last hour's demand (kWh)", min_value=0.0, value=50.0, step=1.0,
                    help="How much energy this zone delivered in the hour just before "
                    "the forecast hour. This is the most recent information available, "
                    "so it is usually the strongest clue.",
                )
                lag_24 = st.number_input(
                    "Demand at this hour yesterday (kWh)", min_value=0.0, value=45.0, step=1.0,
                    help="How much energy this zone delivered in the same hour one day "
                    "earlier. It captures the usual daily rhythm — busy evenings, quiet "
                    "early mornings.",
                )
            with right:
                hour = st.slider(
                    "Forecast hour of day", 0, 23, 18,
                    help="The hour you are predicting, in 24-hour time: 18 means 6 PM.",
                )
                day_of_week = st.selectbox(
                    "Day of week",
                    options=list(range(7)),
                    format_func=lambda d: [
                        "Monday", "Tuesday", "Wednesday", "Thursday",
                        "Friday", "Saturday", "Sunday",
                    ][d],
                    help="Weekdays and weekends usually have different charging patterns.",
                )

            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda m: [
                    "January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December",
                ][m - 1],
                index=8,
                help="Season affects demand; the dataset covers September 2022 to "
                "February 2023.",
            )

            with st.expander("What are these inputs called in the model?"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Form label": "Last hour's demand (kWh)", "Model feature": "lag_1"},
                            {"Form label": "Demand at this hour yesterday (kWh)", "Model feature": "lag_24"},
                            {"Form label": "Forecast hour of day", "Model feature": "hour"},
                            {"Form label": "Day of week", "Model feature": "day_of_week"},
                            {"Form label": "Month", "Model feature": "month"},
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

            submitted = st.button(
                "Predict charging demand", type="primary", use_container_width=True
            )

    if submitted:
        try:
            model = load_model(str(available_models[model_name]))
        except Exception as exc:  # noqa: BLE001 - corrupt/incompatible artefact
            st.error(
                f"Could not load the {model_name} model from "
                f"`{available_models[model_name].relative_to(ROOT)}`: {exc}"
            )
            st.stop()

        feature_values = {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "lag_1": lag_1,
            "lag_24": lag_24,
        }

        # Energy delivered cannot be negative. Reject it here, before any model is
        # loaded or called, and drop the previous result so an old number is never
        # left on screen looking like the answer to this (invalid) request.
        problems = validate_demand_inputs(feature_values)
        if problems:
            st.session_state.pop("latest_prediction", None)
            for message in problems:
                st.error(message)
            st.caption(
                "No prediction was made. Correct the highlighted value(s) and "
                "press **Predict charging demand** again."
            )
            st.stop()

        try:
            expected = list(getattr(model, "feature_names_in_", feature_values.keys()))
            missing = [f for f in expected if f not in feature_values]
            if missing:
                # Honest failure rather than silently substituting a value.
                st.error(
                    "This model expects features the form does not collect: "
                    + ", ".join(missing)
                )
                st.stop()

            # Reject unusable inputs before calling the model, so the message names
            # the field rather than surfacing a raw scikit-learn error.
            bad = [k for k in expected if not np.isfinite(float(feature_values[k]))]
            if bad:
                st.error(
                    "These inputs are not valid numbers: " + ", ".join(bad)
                    + ". Enter a demand value of 0 or more kWh."
                )
                st.stop()

            features = pd.DataFrame(
                [{f: feature_values[f] for f in expected}]
            )
            prediction = float(model.predict(features)[0])
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"Prediction failed for {model_name}: {exc}. If this keeps happening, "
                "retrain the models with `python -m src.models.train_models`."
            )
            st.stop()

        if not np.isfinite(prediction):
            st.error(
                "The model returned an invalid (non-finite) prediction, so no result "
                "is shown. Try different inputs."
            )
            # Drop any previous value so a stale number is never shown as this run.
            st.session_state.pop("latest_prediction", None)
        else:
            st.session_state["latest_prediction"] = {
                "Model": model_name,
                "Predicted demand (kWh)": prediction,
                **feature_values,
            }

    with result_col:
        with st.container(border=True):
            st.markdown("##### Prediction result")
            if "latest_prediction" not in st.session_state:
                st.caption(
                    "No prediction yet. Choose a model and inputs on the left, then "
                    "press **Predict charging demand**."
                )
                st.metric("Predicted energy — next hour", "— kWh")
            else:
                result = st.session_state["latest_prediction"]
                value = result["Predicted demand (kWh)"]
                st.metric("Predicted energy — next hour", f"{value:,.2f} kWh")
                st.caption(
                    f"**{value:,.2f} kilowatt-hours** for this zone during the selected "
                    f"hour. Model used: **{result['Model']}**."
                )

                # Context, derived only from already-loaded results — no new claims.
                context = column_context(metric_lookup, result["Model"])
                if context is not None:
                    st.caption(
                        f"On the test set this model's typical error was "
                        f"**±{context['mae']:.2f} kWh**, so an estimate in the "
                        f"±{context['mae']:.2f} kWh range of the true value would be "
                        f"normal. That is an average over all test rows — individual "
                        f"hours can miss by much more."
                    )

                if value < 0:
                    st.warning(
                        "The model returned a negative estimate. Energy delivered cannot "
                        "be negative, so treat this as a statistical artefact of the "
                        "regression rather than a physical forecast."
                    )
                with st.expander("Inputs used for this prediction"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Last hour's demand (kWh)": result["lag_1"],
                                    "Demand 24 h earlier (kWh)": result["lag_24"],
                                    "Hour of day": f"{result['hour']:02d}:00",
                                    "Day of week": DAY_NAMES[int(result["day_of_week"])],
                                    "Month": MONTH_NAMES[int(result["month"]) - 1],
                                }
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                st.download_button(
                    "Download this prediction (CSV)",
                    data=pd.DataFrame([result]).to_csv(index=False).encode("utf-8"),
                    file_name="latest_ev_demand_prediction.csv",
                    mime="text/csv",
                    key="download_latest_prediction",
                    use_container_width=True,
                )
            st.divider()
            st.caption(
                "This is **one** what-if estimate for the inputs you chose. The "
                "measured accuracy of the models on the test period is reported "
                "separately in section 2."
            )

# =====================================================================
# 2. Model comparison (baseline + both models)
# =====================================================================
st.divider()
st.header("2. Model comparison")
st.caption(
    "These are **measured results**: every model was scored on the same chronological "
    "test rows it never saw during training. They describe how accurate each model is "
    "in general — they are not a single prediction. To try one forecast, use section 1."
)

if comparison_df.empty:
    st.info(
        "Comparison results not found. Run the evaluation pipeline "
        "(`baseline_persistence` → `train_models`) to generate them."
    )
else:
    best_mae = comparison_df.loc[comparison_df["MAE (kWh)"].idxmin()]
    best_rmse = comparison_df.loc[comparison_df["RMSE (kWh)"].idxmin()]
    best_r2 = comparison_df.loc[comparison_df["R²"].idxmax()]
    cards = st.columns(3)
    cards[0].metric(
        "Best MAE — average miss",
        format_kwh(best_mae["MAE (kWh)"]),
        help=f"Lowest mean absolute error: {best_mae['Model']}. "
        "On a typical test hour this model is off by about this many kWh.",
        border=True,
    )
    cards[1].metric(
        "Best RMSE — penalises big misses",
        format_kwh(best_rmse["RMSE (kWh)"]),
        help=f"Lowest root mean squared error: {best_rmse['Model']}. "
        "Same kWh units as MAE, but large errors weigh more.",
        border=True,
    )
    cards[2].metric(
        "Best R² — variation explained",
        f"{best_r2['R²']:.4f}",
        help=f"Highest coefficient of determination: {best_r2['Model']}. "
        "1.0 would be perfect; 0.0 is no better than always predicting the average.",
        border=True,
    )

    st.subheader("What the numbers say")
    for line in explain_comparison(
        comparison_df,
        best_mae={"model": best_mae["Model"], "value": float(best_mae["MAE (kWh)"])},
        best_r2={"model": best_r2["Model"], "value": float(best_r2["R²"])},
    ):
        st.markdown(f"- {line}")

    # Show each trained model against the persistence rule it has to beat.
    gains = [
        (name, improvement_vs_baseline(comparison_df, name))
        for name in ("Linear Regression", "Random Forest")
        if name in set(comparison_df["Model"])
    ]
    gains = [(n, g) for n, g in gains if g is not None]
    if gains:
        st.markdown("**Versus the persistence baseline** (repeat last hour's demand)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": name,
                        "Baseline MAE (kWh)": round(g["baseline_mae"], 4),
                        "Model MAE (kWh)": round(g["model_mae"], 4),
                        "Change in average error": f"{g['mae_pct']:+.1f}%",
                    }
                    for name, g in gains
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "A negative change means the trained model's average error is lower "
            "than simply repeating the previous hour's demand."
        )

    display = comparison_df.copy()
    for column in ("MAE (kWh)", "RMSE (kWh)", "R²"):
        display[column] = display[column].map(lambda v: round(v, 4))
    # The comparison CSV has no row count per model; show a dash, not "None".
    if "Test rows" in display.columns:
        display["Test rows"] = display["Test rows"].map(
            lambda v: f"{int(v):,}" if pd.notna(v) else "—"
        )
    st.markdown("**Full results table**")
    st.dataframe(display, width="stretch", hide_index=True)

    with st.expander("What do MAE, RMSE and R² mean?", expanded=False):
        st.markdown(
            "All three metrics are computed on the **same chronological test rows**, "
            "so the rows can be compared directly."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": name,
                        "What it means": describe_metric(name),
                        "Better": better,
                    }
                    for name, better in (
                        ("MAE (kWh)", "Lower"),
                        ("RMSE (kWh)", "Lower"),
                        ("R²", "Higher"),
                    )
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            "**In practice:**\n\n"
            "* **MAE** is the easiest to read — it is in kWh, the same units as demand, "
            "so it tells you the size of a typical miss.\n"
            "* **RMSE** is always at least as large as MAE. When it is much larger, a "
            "few hours are missed badly even if most are close.\n"
            "* **R²** is unitless. It says how much of the day-to-day variation the model "
            "reproduces, not how large the error is in kWh.\n\n"
            "A model is only described here as better overall when it improves **all "
            "three** metrics over the persistence baseline."
        )

    error_metrics = ["MAE (kWh)", "RMSE (kWh)"]
    st.subheader("Error metric comparison (lower is better)")
    st.bar_chart(comparison_df.set_index("Model")[error_metrics], height=320)
    st.caption(
        "Bars are the same values as the table above. RMSE bars are much taller "
        "because a small number of very high-demand hours carry very large errors."
    )

    st.download_button(
        "Download model comparison (CSV)",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        key="download_comparison",
    )

# =====================================================================
# 3. Saved test predictions: actual vs predicted + error analysis
# =====================================================================
st.divider()
st.header("3. Test-set results (saved predictions)")
st.caption(
    "Stored predictions for the chronological test window — the same rows every "
    "model was scored on. Nothing here is refitted or recomputed."
)

predictions_path = PREDICTIONS_DIR / "model_test_predictions.csv"
predictions = load_csv(str(predictions_path), parse_time=True)

if predictions is None:
    st.info(
        "Saved test predictions not found (`results/predictions/model_test_predictions.csv`). "
        "Run `python -m src.models.train_models` to generate them."
    )
else:
    predictions[ZONE] = predictions[ZONE].astype(str)
    st.caption(
        f"{len(predictions):,} test rows across {predictions[ZONE].nunique()} zones. "
        "These are stored results — nothing is recomputed."
    )

    pred_cols = {
        "Persistence baseline (lag_1)": "baseline_predicted_kwh",
        "Linear Regression": "linear_regression_predicted_kwh",
        "Random Forest": "random_forest_predicted_kwh",
    }
    pred_cols = {k: v for k, v in pred_cols.items() if v in predictions.columns}

    if not pred_cols:
        st.warning("The prediction file has no recognised model prediction columns.")
    else:
        tab_series, tab_error = st.tabs(["Actual vs predicted", "Error analysis"])

        with tab_series:
            pick_left, pick_right = st.columns(2)
            with pick_left:
                model_pick = st.selectbox(
                    "Model", options=list(pred_cols), key="series_model"
                )
            col = pred_cols[model_pick]

            with pick_right:
                zone_options = ["All zones (average per hour)"] + sorted(
                    predictions[ZONE].unique(), key=lambda z: (len(z), z)
                )
                zone_pick = st.selectbox(
                    "Zone", options=zone_options, key="series_zone"
                )

            frame = predictions[["time", ZONE, TARGET, col]].dropna()
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
            frame = frame.dropna()

            if zone_pick != zone_options[0]:
                frame = frame[frame[ZONE] == zone_pick]
                series = (
                    frame.set_index("time")[[TARGET, col]]
                    .rename(columns={TARGET: "Actual (kWh)", col: "Predicted (kWh)"})
                    .sort_index()
                )
                title_suffix = f"zone {zone_pick}"
            else:
                series = (
                    frame.groupby("time")[[TARGET, col]]
                    .mean()
                    .rename(columns={TARGET: "Actual (kWh)", col: "Predicted (kWh)"})
                    .sort_index()
                )
                title_suffix = "all zones averaged"

            if series.empty:
                st.info("No rows available for this selection.")
            else:
                actual, predicted = series["Actual (kWh)"], series["Predicted (kWh)"]
                summary = st.columns(3)
                summary[0].metric(
                    "Mean actual demand",
                    f"{actual.mean():,.2f} kWh",
                    border=True,
                )
                summary[1].metric(
                    "Mean predicted demand",
                    f"{predicted.mean():,.2f} kWh",
                    border=True,
                )
                summary[2].metric(
                    "Mean absolute error",
                    f"{(actual - predicted).abs().mean():,.2f} kWh",
                    border=True,
                    help="Computed over the plotted selection (one zone or all "
                    "zones averaged), so it can differ from the overall test MAE.",
                )
                st.line_chart(series, height=380)
                st.caption(
                    f"{model_pick} — hourly series, {title_suffix}. "
                    "Showing the full saved test window (no sampling)."
                )

        with tab_error:
            err = error_table(predictions)
            err_cols = [c for c in pred_cols if c in err.columns]
            overall = err[err_cols].mean().rename("Mean absolute error (kWh)")
            st.markdown("**Mean absolute error by model (kWh)**")
            st.dataframe(
                overall.to_frame().reset_index().rename(columns={"index": "Model"}),
                width="stretch",
                hide_index=True,
            )

            zone_left, zone_right = st.columns([1, 2], gap="large")
            with zone_left:
                zone_pick2 = st.selectbox(
                    "Per-zone performance (worst 15)",
                    options=err_cols,
                    key="zone_err_model",
                )
                per_zone = (
                    err.groupby(ZONE)[zone_pick2]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                    .rename("MAE (kWh)")
                    .to_frame()
                )
                st.markdown(f"**Highest-error zones — {zone_pick2}**")
                st.dataframe(per_zone, width="stretch")

            with zone_right:
                st.markdown("**Error by demand level**")
                # Same deciles as the generated figure, so the two never disagree.
                deciles = pd.qcut(
                    err[TARGET].rank(method="first"), 10,
                    labels=[f"D{i}" for i in range(1, 11)],
                )
                by_level = err.groupby(deciles, observed=True)[err_cols].mean()
                st.bar_chart(by_level, height=320)
                st.caption(
                    "Average absolute error within each tenth of actual demand "
                    "(D1 = lowest, D10 = highest). Large errors concentrate in the "
                    "highest-demand hours, so a single MAE understates the "
                    "uncertainty at peak load."
                )

        st.download_button(
            "Download saved test predictions (CSV)",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="model_test_predictions.csv",
            mime="text/csv",
            key="download_test_predictions",
        )

# =====================================================================
# 4. Validation study (optional) + generated figures
# =====================================================================
study = load_json(str(EXPERIMENTS_DIR / "validation_study.json"))
if study:
    st.divider()
    st.header("4. Validation study (model selection)")
    design = study.get("design", {})
    fold = design.get("validation_fold", {})
    st.caption(
        "Candidates were compared on a validation window carved from the end of the "
        "training period. The final test period was scored **once**, with the single "
        "candidate selected on validation."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Validation rows", f"{fold.get('valid_rows', 0):,}", border=True)
    c2.metric("Search-train rows", f"{fold.get('train_rows', 0):,}", border=True)
    c3.metric("Selection rule", "Lowest val. RMSE", border=True)

    st.subheader("Validation-set comparison")
    val_rows = [
        {
            "Candidate": r["label"],
            "Val MAE (kWh)": round(r["valid"]["mae"], 4),
            "Val RMSE (kWh)": round(r["valid"]["rmse"], 4),
            "Val R²": round(r["valid"]["r2"], 4),
        }
        for r in study.get("validation_results", [])
    ]
    if val_rows:
        st.dataframe(pd.DataFrame(val_rows), width="stretch", hide_index=True)

    final = study.get("final_test", {})
    selected = study.get("selected_candidate", {})
    verification = study.get("verdict", {})
    if final:
        st.subheader("Final test result (scored once)")
        st.markdown(f"**Selected on validation:** {selected.get('label', 'n/a')}")
        f1, f2, f3 = st.columns(3)
        f1.metric(
            "Final-test MAE", f"{final.get('mae', float('nan')):.2f} kWh", border=True
        )
        f2.metric(
            "Final-test RMSE", f"{final.get('rmse', float('nan')):.2f} kWh", border=True
        )
        f3.metric(
            "Final-test R²", f"{final.get('r2', float('nan')):.4f}", border=True
        )
        if "beats_reference_on_all_three" in verification:
            if verification["beats_reference_on_all_three"]:
                st.success(
                    "This candidate beats the reference Random Forest on all three "
                    f"test metrics (ΔMAE {verification['delta_mae']:+.2f} kWh)."
                )
            else:
                st.info(
                    "This candidate does **not** beat the reference Random Forest on "
                    "all three test metrics, so the reference model remains the "
                    "reported result."
                )
        st.caption(
            "Validation metrics are not test metrics: the validation numbers above "
            "describe the selection window only."
        )

figure_specs = [
    ("fig_actual_vs_predicted.png", "Actual vs predicted demand"),
    ("fig_error_distribution.png", "Prediction error distribution"),
    ("fig_model_comparison.png", "Model performance comparison"),
    ("fig_zone_error.png", "Per-zone error (worst zones)"),
    ("fig_high_demand_errors.png", "Error by demand level"),
    ("fig_hourly_demand_profile.png", "Hourly demand profile"),
    ("fig_daily_demand_profile.png", "Daily demand patterns"),
]
existing_figures = [(p, t) for p, t in figure_specs if (FIGURES_DIR / p).is_file()]
if existing_figures:
    st.divider()
    st.header("5. Generated figures")
    st.caption("Produced by `python -m src.visualization.plots` — regenerate there.")
    fig_left, fig_right = st.columns(2, gap="large")

with fig_left:
    st.subheader("Hourly demand profile")
    hourly_path = FIGURES_DIR / "fig_hourly_demand_profile.png"
    if hourly_path.is_file():
        st.image(str(hourly_path), width="stretch")
        st.caption(
            "Mean demand per zone-hour by hour of day over the full dataset, with "
            "the test window overlaid. Shows the daily cycle the models exploit "
            "through `hour` and `lag_24`."
        )
    else:
        st.info(
            "Hourly profile not available. Run `python -m src.visualization.plots` "
            "(needs `data/processed/demand_long.csv`)."
        )

with fig_right:
    st.subheader("Daily demand patterns")
    daily_path = FIGURES_DIR / "fig_daily_demand_profile.png"
    if daily_path.is_file():
        st.image(str(daily_path), width="stretch")
        st.caption(
            "Mean demand by day of week, and the hourly shape on weekdays versus "
            "the weekend — the reason `day_of_week` is a predictor."
        )
    else:
        st.info(
            "Daily profile not available. Run `python -m src.visualization.plots` "
            "(needs `data/processed/demand_long.csv`)."
        )

remaining_figures = [(p, t) for p, t in existing_figures if p not in {
    "fig_hourly_demand_profile.png", "fig_daily_demand_profile.png"
}]
if remaining_figures:
    fig_cols = st.columns(2, gap="large")
    for i, (name, title) in enumerate(remaining_figures):
        with fig_cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.image(str(FIGURES_DIR / name), width="stretch")

# =====================================================================
# 6. Models and limitations
# =====================================================================
st.divider()
st.header("6. Models and limitations")
st.caption(
    "What each approach does, and what this dashboard deliberately does not claim."
)

model_col, limit_col = st.columns(2, gap="large")

with model_col:
    st.subheader("Models")
    with st.container(border=True):
        st.markdown("**Linear Regression**")
        st.write(
            "Ordinary least squares over the calendar features and two demand lags. "
            "Deterministic and interpretable, but it cannot represent non-linear "
            "interactions between hour, day and recent demand."
        )
    with st.container(border=True):
        st.markdown("**Random Forest**")
        st.write(
            "100 regression trees (max depth 20, min samples per leaf 20, fixed seed). "
            "Captures non-linear structure and is the best-performing model verified on "
            "the test set."
        )
    with st.container(border=True):
        st.markdown("**Persistence baseline**")
        st.write(
            "Predicts the next hour using the previous hour's demand (`y_pred = lag_1`). "
            "No model is fitted. It is the reference every model must beat on identical "
            "test rows."
        )

with limit_col:
    st.subheader("Limitations")
    with st.container(border=True):
        st.markdown(
            "* Only two lags (`lag_1`, `lag_24`) plus calendar features and static zone "
            "attributes — no weather, holidays or events.\n"
            "* The dataset's occupancy/duration files are same-hour observations and are "
            "excluded as predictors to avoid target leakage.\n"
            "* Errors are concentrated in a handful of very high-demand zones, so per-zone "
            "forecasts there carry much larger uncertainty than the overall MAE suggests.\n"
            "* The manual form is model-agnostic and does not adjust for zone characteristics; "
            "it is a demonstration of the saved estimators, not a deployable per-zone service."
        )

st.divider()
st.caption("B.Tech project | EV Charging Demand Prediction Using Machine Learning")

