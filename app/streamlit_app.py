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

    /* Streamlit 1.63's default accent is red (#FF4B4B), and `--primary-color`
       does not reach the slider. The styled components in Slider.js give their
       names away through the emotion target ids, so match those rather than
       the DOM shape: efbyxod3 = 12px round thumb, efbyxod5 = 4px track bar
       (grey rail + filled part), efbyxod4 = the floating value label. */
    [class*="efbyxod3"] { background-color: #1769AA !important; }
    [class*="efbyxod4"],
    [data-testid="stSliderThumbValue"],
    [data-testid="stSliderThumbValue"] * {
        color: #1769AA !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }
    /* Track: keep the grey rail on the right, paint only the filled part. */
    [class*="efbyxod5"] {
        background: linear-gradient(
            to right,
            #1769AA 0%,
            #1769AA 0.36%,
            #D8E5F2 0.36%,
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

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #D8E5F2;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(22, 58, 99, 0.06);
    }
    [data-testid="stMetricLabel"] { color: #526579 !important; }
    [data-testid="stMetricValue"] { color: #1769AA !important; font-weight: 700 !important; }
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

if not available_models:
    st.error(
        "No trained models found in `models/`. Train them first with "
        "`python -m src.models.train_models`."
    )
else:
    if missing_models:
        st.warning("Model artefact(s) not found: " + ", ".join(missing_models))

    # Show each model's verified test performance next to its selector entry.
    metric_lookup = (
        comparison_df.set_index("Model") if not comparison_df.empty else pd.DataFrame()
    )

    def _label(name: str) -> str:
        if metric_lookup.empty or name not in metric_lookup.index:
            return name
        row = metric_lookup.loc[name]
        return f"{name} — test MAE {row['MAE (kWh)']:.2f} kWh, R² {row['R²']:.4f}"

    model_name = st.selectbox(
        "Choose a trained model", options=list(available_models.keys()), format_func=_label
    )

    left, right = st.columns(2)
    with left:
        lag_1 = st.number_input(
            "Demand one hour ago (kWh)", min_value=0.0, value=50.0, step=1.0,
            help="Most recent observation available at forecast time (t-1).",
        )
        lag_24 = st.number_input(
            "Demand 24 hours ago (kWh)", min_value=0.0, value=45.0, step=1.0,
            help="Same hour on the previous day — captures the daily cycle.",
        )
        hour = st.slider("Hour of day", 0, 23, 18)
    with right:
        day_of_week = st.selectbox(
            "Day of week",
            options=list(range(7)),
            format_func=lambda d: [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ][d],
        )
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda m: [
                "January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December",
            ][m - 1],
            index=8,
        )

    if st.button("Predict charging demand", type="primary"):
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
            features = pd.DataFrame(
                [{f: feature_values[f] for f in expected}]
            )
            prediction = float(model.predict(features)[0])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            st.stop()

        if not np.isfinite(prediction):
            st.error("The model returned an invalid (non-finite) prediction.")
        else:
            st.session_state["latest_prediction"] = {
                "Model": model_name,
                "Predicted demand (kWh)": prediction,
                **feature_values,
            }

    if "latest_prediction" in st.session_state:
        result = st.session_state["latest_prediction"]
        st.subheader("Prediction result")
        st.metric(
            "Predicted charging demand",
            f"{result['Predicted demand (kWh)']:,.2f} kWh",
        )
        st.caption(f"Model: {result['Model']}")
        if result["Predicted demand (kWh)"] < 0:
            st.warning(
                "The model returned a negative estimate. Treat it as a statistical "
                "artefact of the regression, not as a physical forecast."
            )
        st.download_button(
            "Download this prediction (CSV)",
            data=pd.DataFrame([result]).to_csv(index=False).encode("utf-8"),
            file_name="latest_ev_demand_prediction.csv",
            mime="text/csv",
            key="download_latest_prediction",
        )

# =====================================================================
# 2. Model comparison (baseline + both models)
# =====================================================================
st.divider()
st.header("2. Model comparison")

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
    cards[0].metric("Best MAE", f"{best_mae['MAE (kWh)']:.2f} kWh", help=best_mae["Model"])
    cards[1].metric("Best RMSE", f"{best_rmse['RMSE (kWh)']:.2f} kWh", help=best_rmse["Model"])
    cards[2].metric("Best R²", f"{best_r2['R²']:.4f}", help=best_r2["Model"])

    display = comparison_df.copy()
    display["MAE (kWh)"] = display["MAE (kWh)"].map(lambda v: round(v, 4))
    display["RMSE (kWh)"] = display["RMSE (kWh)"].map(lambda v: round(v, 4))
    display["R²"] = display["R²"].map(lambda v: round(v, 4))
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown(
        "**How to read these metrics** (all on the same chronological test rows):\n\n"
        "* **MAE** — mean absolute error in kWh. The typical size of a miss; "
        "directly comparable to demand values.\n"
        "* **RMSE** — root mean squared error in kWh. Same units, but large "
        "misses count more. RMSE ≫ MAE means a few very large errors.\n"
        "* **R²** — share of the test-set variance explained. 1.0 is perfect, "
        "0.0 is no better than predicting the mean.\n\n"
        "A model is only described as better when it improves **all three** "
        "metrics over the persistence baseline."
    )

    error_metrics = ["MAE (kWh)", "RMSE (kWh)"]
    st.subheader("Error metric comparison (lower is better)")
    st.bar_chart(comparison_df.set_index("Model")[error_metrics])

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
            model_pick = st.selectbox(
                "Model", options=list(pred_cols), key="series_model"
            )
            col = pred_cols[model_pick]

            zone_options = ["All zones (average per hour)"] + sorted(
                predictions[ZONE].unique(), key=lambda z: (len(z), z)
            )
            zone_pick = st.selectbox("Zone", options=zone_options, key="series_zone")

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

            zone_pick2 = st.selectbox(
                "Per-zone performance (worst 15)", options=err_cols, key="zone_err_model"
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

            st.markdown("**Error by demand level**")
            deciles = pd.qcut(
                err[TARGET].rank(method="first"), 5, labels=["D1 (lowest)", "D2", "D3", "D4", "D5 (highest)"]
            )
            by_level = err.groupby(deciles, observed=True)[err_cols].mean()
            st.bar_chart(by_level, height=320)
            st.caption(
                "Average absolute error within each fifth of actual demand. Large "
                "errors concentrate in the highest-demand hours, so a single MAE "
                "understates the uncertainty at peak load."
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
    c1.metric("Validation rows", f"{fold.get('valid_rows', 0):,}")
    c2.metric("Search-train rows", f"{fold.get('train_rows', 0):,}")
    c3.metric("Selection rule", "Lowest val. RMSE")

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
        st.markdown(f"**Selected on validation:** {selected.get('label', 'n/a')}")
        f1, f2, f3 = st.columns(3)
        f1.metric("Final-test MAE", f"{final.get('mae', float('nan')):.2f} kWh")
        f2.metric("Final-test RMSE", f"{final.get('rmse', float('nan')):.2f} kWh")
        f3.metric("Final-test R²", f"{final.get('r2', float('nan')):.4f}")
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
]
existing_figures = [(p, t) for p, t in figure_specs if (FIGURES_DIR / p).is_file()]
if existing_figures:
    st.divider()
    st.header("5. Generated figures")
    st.caption("Produced by `python -m src.visualization.plots` — regenerate there.")
    for name, title in existing_figures:
        with st.expander(title, expanded=False):
            st.image(str(FIGURES_DIR / name), width="stretch")

# =====================================================================
# 6. Models and limitations
# =====================================================================
st.divider()
st.header("6. Models and limitations")

with st.expander("Linear Regression"):
    st.write(
        "Ordinary least squares over the calendar features and two demand lags. "
        "Deterministic and interpretable, but it cannot represent non-linear "
        "interactions between hour, day and recent demand."
    )
with st.expander("Random Forest"):
    st.write(
        "100 regression trees (max depth 20, min samples per leaf 20, fixed seed). "
        "Captures non-linear structure and is the best-performing model verified on "
        "the test set."
    )
with st.expander("Persistence baseline"):
    st.write(
        "Predicts the next hour using the previous hour's demand (`y_pred = lag_1`). "
        "No model is fitted. It is the reference every model must beat on identical "
        "test rows."
    )

st.markdown(
    "**Limitations**\n\n"
    "* Only two lags (`lag_1`, `lag_24`) plus calendar features and static zone "
    "attributes — no weather, holidays or events.\n"
    "* The dataset's occupancy/duration files are same-hour observations and are "
    "excluded as predictors to avoid target leakage.\n"
    "* Errors are concentrated in a handful of very high-demand zones, so per-zone "
    "forecasts there carry much larger uncertainty than the overall MAE suggests.\n"
    "* The manual form is model-agnostic and does not adjust for zone characteristics; "
    "it is a demonstration of the saved estimators, not a deployable per-zone service."
)

st.caption("B.Tech project | EV Charging Demand Prediction Using Machine Learning")

