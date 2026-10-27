
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project paths and page configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = ROOT / "results" / "tables"
PREDICTIONS_DIR = ROOT / "results" / "predictions"

st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------
# 2. Page heading
# --------------------------------------------------
st.title("⚡ EV Charging Demand Prediction")
st.caption("One-hour-ahead demand forecasting using Machine Learning")

st.markdown(
    """
    This application predicts hourly EV charging demand using historical
    demand and calendar features. Choose a trained model, enter the known
    input values, and generate a prediction.
    """
)

# --------------------------------------------------
# 3. Load trained models
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


model_files = {
    "Linear Regression": MODELS_DIR / "linear_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
}

available_models = {
    name: path
    for name, path in model_files.items()
    if path.is_file()
}

if not available_models:
    st.error(
        "No trained model files were found. "
        "Please check the models/ folder."
    )
    st.stop()

# --------------------------------------------------
# 4. Prediction inputs
# --------------------------------------------------
st.sidebar.header("Prediction settings")

model_name = st.sidebar.selectbox(
    "Choose a model",
    options=list(available_models.keys()),
    key="prediction_model_selector",
)

st.subheader("Enter prediction features")

col1, col2 = st.columns(2)

with col1:
    lag_1 = st.number_input(
        "Demand 1 hour ago (kWh)",
        min_value=0.0,
        value=50.0,
        step=1.0,
        key="input_lag_1",
    )

    lag_24 = st.number_input(
        "Demand 24 hours ago (kWh)",
        min_value=0.0,
        value=45.0,
        step=1.0,
        key="input_lag_24",
    )

    hour = st.slider(
        "Hour of day",
        min_value=0,
        max_value=23,
        value=18,
        key="input_hour",
    )

with col2:
    day_of_week = st.selectbox(
        "Day of week (0 = Monday)",
        options=list(range(7)),
        format_func=lambda day: [
            "Monday (0)",
            "Tuesday (1)",
            "Wednesday (2)",
            "Thursday (3)",
            "Friday (4)",
            "Saturday (5)",
            "Sunday (6)",
        ][day],
        key="input_day_of_week",
    )

    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda value: [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December",
        ][value - 1],
        index=8,
        key="input_month",
    )

# --------------------------------------------------
# 5. Make a prediction
# --------------------------------------------------
if st.button(
    "Predict charging demand",
    type="primary",
    key="run_prediction",
):
    try:
        model = load_model(available_models[model_name])

        # The values here must match the model's training features.
        feature_values = {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "lag_1": lag_1,
            "lag_24": lag_24,
        }

        # Follow the saved model's exact feature order if available.
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            missing_features = [
                name for name in expected_features
                if name not in feature_values
            ]

            if missing_features:
                st.error(
                    "The model requires unsupported features: "
                    + ", ".join(missing_features)
                )
                st.stop()

            features = pd.DataFrame(
                [[feature_values[name] for name in expected_features]],
                columns=expected_features,
            )
        else:
            features = pd.DataFrame([feature_values])

        prediction = float(model.predict(features)[0])

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")
        else:
            st.success(
                f"Predicted charging demand: {prediction:,.2f} kWh"
            )
            st.caption(f"Selected model: {model_name}")

            if prediction < 0:
                st.warning(
                    "The model predicted negative demand. "
                    "Check the inputs and model behaviour."
                )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# 6. Model performance comparison
# --------------------------------------------------
st.divider()
st.header("Model Performance Comparison")

comparison_file = TABLES_DIR / "model_comparison.csv"

if comparison_file.is_file():
    try:
        comparison = pd.read_csv(comparison_file)
        comparison.columns = [
            str(column).strip() for column in comparison.columns
        ]

        # Support the column names used by the generated comparison CSV.
        rename_map = {
            "model": "Model",
            "split": "Split",
            "mae": "MAE (kWh)",
            "rmse": "RMSE (kWh)",
            "r2": "R²",
            "beats_baseline": "Beats baseline",
        }
        comparison = comparison.rename(columns=rename_map)

        # Use test-set metrics only when split is provided.
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
            st.info("No test-set comparison results were found.")
        else:
            st.caption(
                "Lower MAE and RMSE are better. Higher R² is better."
            )

            display_columns = [
                column for column in [
                    "Model",
                    "MAE (kWh)",
                    "RMSE (kWh)",
                    "R²",
                    "Beats baseline",
                ]
                if column in comparison.columns
            ]

            st.dataframe(
                comparison[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            if "MAE (kWh)" in comparison.columns:
                valid_rows = comparison.dropna(subset=["MAE (kWh)"])

                if not valid_rows.empty:
                    best_row = valid_rows.loc[
                        valid_rows["MAE (kWh)"].idxmin()
                    ]
                    st.success(
                        f"Lowest test MAE in the comparison: "
                        f"{best_row.get('Model', 'Unknown model')} "
                        f"({best_row['MAE (kWh)']:.2f} kWh)"
                    )

            chart_metrics = [
                metric for metric in ["MAE (kWh)", "RMSE (kWh)"]
                if metric in comparison.columns
            ]

            if "Model" in comparison.columns and chart_metrics:
                st.subheader("Prediction error comparison")

                chart_data = comparison.set_index("Model")[chart_metrics]
                st.bar_chart(chart_data)

    except Exception as exc:
        st.warning(f"Could not load comparison results: {exc}")
else:
    st.info(
        "The comparison file was not found at "
        "results/tables/model_comparison.csv."
    )

# --------------------------------------------------
# 7. Actual versus predicted demand
# --------------------------------------------------
st.divider()
st.header("Actual vs Predicted Demand")

prediction_file = (
    PREDICTIONS_DIR / "model_test_predictions.csv"
)

if prediction_file.is_file():
    try:
        predictions = pd.read_csv(prediction_file)

        st.caption(
            "This chart uses saved test-set predictions, not the manual "
            "inputs from the prediction form."
        )

        # Identify compatible columns in the saved prediction file.
        columns = list(predictions.columns)

        target_candidates = [
            "demand_kwh",
            "actual_kwh",
            "actual",
            "y_true",
        ]
        actual_column = next(
            (name for name in target_candidates if name in columns),
            None,
        )

        time_column = next(
            (name for name in ["time", "timestamp", "datetime"]
             if name in columns),
            None,
        )

        if actual_column is None:
            st.info(
                "The saved prediction file does not contain a recognised "
                "actual-demand column, so this chart cannot be displayed."
            )
        else:
            model_prediction_columns = [
                column for column in columns
                if (
                    "linear_regression" in column.lower()
                    or "random_forest" in column.lower()
                )
                and (
                    "predicted" in column.lower()
                    or "prediction" in column.lower
                )
            ]

            # Also support a long-format prediction file.
            model_name_column = next(
                (name for name in ["model", "model_name"]
                 if name in columns),
                None,
            )

            generic_prediction_column = next(
                (name for name in [
                    "predicted_kwh",
                    "prediction",
                    "y_pred",
                ] if name in columns),
                None,
            )

            if model_prediction_columns:
                selected_chart_model = st.selectbox(
                    "Model to display in the chart",
                    options=model_prediction_columns,
                    key="actual_predicted_model_selector",
                )

                chart_frame = predictions.copy()
                chart_frame[actual_column] = pd.to_numeric(
                    chart_frame[actual_column], errors="coerce"
                )
                chart_frame[selected_chart_model] = pd.to_numeric(
                    chart_frame[selected_chart_model], errors="coerce"
                )

                chart_frame = chart_frame.dropna(
                    subset=[actual_column, selected_chart_model]
                )

                if time_column:
                    chart_frame[time_column] = pd.to_datetime(
                        chart_frame[time_column], errors="coerce"
                    )
                    chart_frame = chart_frame.sort_values(time_column)

                chart_frame = chart_frame.head(200)

                if not chart_frame.empty:
                    chart_frame = chart_frame.reset_index(drop=True)
                    chart_frame.index.name = "Test row"

                    chart_data = chart_frame[
                        [actual_column, selected_chart_model]
                    ].rename(
                        columns={
                            actual_column: "Actual demand (kWh)",
                            selected_chart_model: "Predicted demand (kWh)",
                        }
                    )

                    st.line_chart(chart_data)
                    st.caption(
                        "Showing up to 200 test rows for readability."
                    )

            elif (
                generic_prediction_column
                and model_name_column
                and model_name_column in predictions.columns
            ):
                chart_model = st.selectbox(
                    "Model to display in the chart",
                    options=sorted(
                        predictions[model_name_column].dropna().unique()
                    ),
                    key="long_format_model_selector",
                )

                chart_frame = predictions[
                    predictions[model_name_column] == chart_model
                ].copy()

                chart_frame[actual_column] = pd.to_numeric(
                    chart_frame[actual_column], errors="coerce"
                )
                chart_frame[generic_prediction_column] = pd.to_numeric(
                    chart_frame[generic_prediction_column], errors="coerce"
                )

                chart_frame = chart_frame.dropna(
                    subset=[actual_column, generic_prediction_column]
                )

                if time_column:
                    chart_frame[time_column] = pd.to_datetime(
                        chart_frame[time_column], errors="coerce"
                    )
                    chart_frame = chart_frame.sort_values(time_column)

                chart_frame = chart_frame.head(200).reset_index(drop=True)
                chart_frame.index.name = "Test row"

                chart_data = chart_frame[
                    [actual_column, generic_prediction_column]
                ].rename(
                    columns={
                        actual_column: "Actual demand (kWh)",
                        generic_prediction_column: "Predicted demand (kWh)",
                    }
                )

                st.line_chart(chart_data)

            else:
                st.info(
                    "The combined prediction file exists, but its prediction "
                    "column names do not match the supported formats. "
                    "Check results/predictions/model_test_predictions.csv."
                )

    except Exception as exc:
        st.warning(f"Could not display test predictions: {exc}")
else:
    st.info(
        "Saved test predictions were not found. Expected file: "
        "results/predictions/model_test_predictions.csv."
    )

# --------------------------------------------------
# 8. Explain the models
# --------------------------------------------------
st.divider()
st.header("About the Models")

with st.expander("Linear Regression"):
    st.write(
        "Linear Regression learns a mathematical relationship between "
        "historical demand, calendar features, and the target demand."
    )

with st.expander("Random Forest"):
    st.write(
        "Random Forest combines multiple decision trees and can learn "
        "more complex relationships between historical demand and "
        "calendar features."
    )

with st.expander("Persistence baseline"):
    st.write(
        "The persistence baseline predicts that the next hour's demand "
        "will equal the previous hour's observed demand (lag_1). It is "
        "a simple reference for evaluating the machine learning models."
    )

# --------------------------------------------------
# 9. Footer
# --------------------------------------------------
st.caption(
    "B.Tech project | EV Charging Demand Prediction Using Machine Learning"
)