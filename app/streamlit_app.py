
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "results" / "tables"
PREDICTIONS_DIR = ROOT / "results" / "predictions"

st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide",
)

# -------------------- Blue & White UI Theme --------------------
st.markdown("""
<style>
.stApp {
    background-color: #F5F8FC;
    color: #1E293B;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

h1, h2, h3 {
    color: #163A63 !important;
    font-weight: 700 !important;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #1769AA !important;
    color: white !important;
    border: 1px solid #1769AA !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
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

[data-testid="stMetricLabel"] {
    color: #526579 !important;
}

[data-testid="stMetricValue"] {
    color: #1769AA !important;
    font-weight: 700 !important;
}

hr {
    border-color: #D8E5F2 !important;
}
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------------------

# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    div[data-testid="stMetric"] {
        background: #f4f7fb;
        border: 1px solid #e0e7f0;
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ EV Charging Demand Prediction")
st.caption("Machine learning dashboard | One-hour-ahead demand forecasting")
st.markdown(
    "Explore charging-demand predictions, compare model performance, "
    "and download the existing evaluation results."
)

# -------------------- Load models --------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


model_files = {
    "Linear Regression": MODELS_DIR / "linear_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
}

available_models = {
    name: path for name, path in model_files.items()
    if path.is_file()
}

if not available_models:
    st.error("No trained models found in the models folder.")
    st.stop()

# -------------------- Prediction --------------------
st.divider()
st.header("1. Predict Charging Demand")
st.write("Enter the known calendar and historical-demand features.")

model_name = st.selectbox(
    "Choose a trained model",
    options=list(available_models.keys()),
)

left, right = st.columns(2)

with left:
    lag_1 = st.number_input(
        "Demand 1 hour ago (kWh)",
        min_value=0.0, value=50.0, step=1.0,
    )
    lag_24 = st.number_input(
        "Demand 24 hours ago (kWh)",
        min_value=0.0, value=45.0, step=1.0,
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
            "January", "February", "March", "April",
            "May", "June", "July", "August", "September",
            "October", "November", "December",
        ][m - 1],
        index=8,
    )

if st.button("Predict charging demand", type="primary"):
    try:
        model = load_model(available_models[model_name])
        feature_values = {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "lag_1": lag_1,
            "lag_24": lag_24,
        }

        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = [f for f in expected if f not in feature_values]
            if missing:
                st.error("Missing required model features: " + ", ".join(missing))
                st.stop()
            features = pd.DataFrame(
                [[feature_values[f] for f in expected]],
                columns=expected,
            )
        else:
            features = pd.DataFrame([feature_values])

        prediction = float(model.predict(features)[0])

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")
        else:
            st.session_state["latest_prediction"] = {
                "Model": model_name,
                "Predicted demand (kWh)": prediction,
                **feature_values,
            }

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

if "latest_prediction" in st.session_state:
    result = st.session_state["latest_prediction"]
    st.subheader("Prediction result")
    st.metric(
        "Predicted charging demand",
        f"{result['Predicted demand (kWh)']:,.2f} kWh",
    )
    st.caption(f"Model used: {result['Model']}")
    if result["Predicted demand (kWh)"] < 0:
        st.warning("The model returned a negative demand estimate.")

    result_csv = pd.DataFrame([result]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download this prediction (CSV)",
        data=result_csv,
        file_name="latest_ev_demand_prediction.csv",
        mime="text/csv",
        key="download_latest_prediction",
    )

# -------------------- Model comparison --------------------
st.divider()
st.header("2. Model Performance Comparison")

comparison_file = TABLES_DIR / "model_comparison.csv"

if comparison_file.is_file():
    try:
        comparison = pd.read_csv(comparison_file)
        comparison.columns = [str(c).strip() for c in comparison.columns]

        rename_map = {
            "model": "Model",
            "split": "Split",
            "mae": "MAE (kWh)",
            "rmse": "RMSE (kWh)",
            "r2": "R²",
            "beats_baseline": "Beats baseline",
        }
        comparison = comparison.rename(columns=rename_map)

        if "Split" in comparison.columns:
            comparison = comparison[
                comparison["Split"].astype(str).str.lower() == "test"
            ].copy()

        for metric in ["MAE (kWh)", "RMSE (kWh)", "R²"]:
            if metric in comparison.columns:
                comparison[metric] = pd.to_numeric(
                    comparison[metric], errors="coerce"
                )

        if comparison.empty:
            st.info("No test-set comparison rows were found.")
        else:
            # Summary cards show the best value per metric in this table.
            cards = st.columns(3)
            for col, metric, label, mode in [
                (cards[0], "MAE (kWh)", "Best MAE", "min"),
                (cards[1], "RMSE (kWh)", "Best RMSE", "min"),
                (cards[2], "R²", "Best R²", "max"),
            ]:
                if metric in comparison.columns:
                    values = comparison[metric].dropna()
                    if not values.empty:
                        value = values.min() if mode == "min" else values.max()
                        col.metric(label, f"{value:.3f}")
                    else:
                        col.metric(label, "N/A")
                else:
                    col.metric(label, "N/A")

            display_columns = [
                c for c in [
                    "Model", "MAE (kWh)", "RMSE (kWh)",
                    "R²", "Beats baseline",
                ] if c in comparison.columns
            ]

            st.dataframe(
                comparison[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download model comparison (CSV)",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv",
                key="download_comparison",
            )

            error_metrics = [
                c for c in ["MAE (kWh)", "RMSE (kWh)"]
                if c in comparison.columns
            ]
            if "Model" in comparison.columns and error_metrics:
                st.subheader("Error metric comparison")
                st.bar_chart(comparison.set_index("Model")[error_metrics])

    except Exception as exc:
        st.warning(f"Could not load model comparison: {exc}")
else:
    st.info("Comparison file not found: results/tables/model_comparison.csv")

# -------------------- Saved test predictions --------------------
st.divider()
st.header("3. Actual vs Predicted Demand")

prediction_file = PREDICTIONS_DIR / "model_test_predictions.csv"

if prediction_file.is_file():
    try:
        predictions = pd.read_csv(prediction_file)
        st.caption(
            "This chart uses saved test-set results, not the manual prediction form."
        )

        actual_candidates = ["demand_kwh", "actual_kwh", "actual", "y_true"]
        actual_col = next(
            (c for c in actual_candidates if c in predictions.columns), None
        )
        time_col = next(
            (c for c in ["time", "timestamp", "datetime"]
             if c in predictions.columns), None
        )

        pred_cols = [
            c for c in predictions.columns
            if ("linear_regression" in c.lower()
                or "random_forest" in c.lower())
            and ("predicted" in c.lower() or "prediction" in c.lower())
        ]

        if actual_col is None:
            st.info("No recognised actual-demand column was found.")
        elif pred_cols:
            selected_col = st.selectbox(
                "Choose prediction series",
                pred_cols,
                key="saved_prediction_series",
            )
            frame = predictions.copy()
            frame[actual_col] = pd.to_numeric(frame[actual_col], errors="coerce")
            frame[selected_col] = pd.to_numeric(
                frame[selected_col], errors="coerce"
            )
            frame = frame.dropna(subset=[actual_col, selected_col])

            if time_col:
                frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
                frame = frame.sort_values(time_col)

            frame = frame.head(200).reset_index(drop=True)
            if not frame.empty:
                chart = frame[[actual_col, selected_col]].rename(
                    columns={
                        actual_col: "Actual demand (kWh)",
                        selected_col: "Predicted demand (kWh)",
                    }
                )
                st.line_chart(chart)
                st.caption("Showing up to 200 saved test rows.")

        else:
            st.info(
                "The combined prediction file does not contain recognised "
                "model prediction columns."
            )

        st.download_button(
            "Download saved test predictions (CSV)",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="model_test_predictions.csv",
            mime="text/csv",
            key="download_test_predictions",
        )

    except Exception as exc:
        st.warning(f"Could not display saved test predictions: {exc}")
else:
    st.info(
        "Saved predictions not found: results/predictions/model_test_predictions.csv"
    )

# -------------------- Model explanations --------------------
st.divider()
st.header("4. About the Models")

with st.expander("Linear Regression"):
    st.write(
        "Learns a linear relationship between calendar features, recent demand, "
        "and charging demand."
    )

with st.expander("Random Forest"):
    st.write(
        "Combines decision trees to learn more complex relationships between "
        "calendar features, recent demand, and charging demand."
    )

with st.expander("Persistence baseline"):
    st.write(
        "Uses demand from one hour ago (lag_1) as the next-hour estimate. "
        "It provides a simple reference for model evaluation."
    )

st.caption(
    "B.Tech project | EV Charging Demand Prediction Using Machine Learning"
)