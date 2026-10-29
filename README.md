# EV Charging Demand Prediction Using Machine Learning

A machine learning project that predicts electric vehicle (EV) charging demand —
the expected charging load at a given station or region over a future time window —
using historical charging session records and contextual data (time of day, day of
week, weather, station attributes, and traffic/usage patterns).

The goal is to produce a reproducible pipeline that turns raw session logs into a
validated forecasting model, with all results and metrics saved to disk so that
every experiment can be audited and compared.

## Project status

Pipeline complete through trained models and an interactive dashboard. The UrbanEV
raw dataset is in place under `data/raw/`; validation, preprocessing, lag-feature
generation, a model-free persistence baseline, **two trained regressors (Linear
Regression and Random Forest)**, and a **Streamlit application** in `app/` are all
implemented and verified. Both trained models beat the persistence baseline on the
same test rows.

### Implemented and verified

| Stage | Module | Output |
| --- | --- | --- |
| Dataset validation (read-only) | `src/data/validate_dataset.py` | `reports/dataset_validation.txt` |
| Preprocessing (wide → long, calendar features, chronological split) | `src/preprocessing/prepare_data.py` | `data/processed/demand_long.csv`, `train.csv`, `test.csv`, `split_metadata.csv`, `split_summary.json` |
| Lag features (`lag_1`, `lag_24`, leakage-safe, per zone) | `src/features/demand_lags.py` | `data/processed/features_lag.csv`, `features_train.csv`, `features_test.csv`, `features_summary.json` |
| Persistence baseline (`y_pred = lag_1`) | `src/evaluation/baseline_persistence.py` | `data/processed/baseline_test_predictions.csv`, `reports/baseline_metrics.json`, `reports/baseline_persistence.md` |
| Model training and evaluation (Linear Regression, Random Forest) | `src/models/train_models.py` | `models/linear_regression.joblib`, `models/random_forest.joblib`, `reports/model_metrics.json`, `reports/model_comparison.md`, `results/tables/model_comparison.csv`, `results/predictions/*` |
| Interactive dashboard | `app/streamlit_app.py` | Streamlit UI: manual prediction, model comparison, actual-vs-predicted chart, CSV downloads |

### Results

All numbers are on the **same chronological test rows** (238,975 rows, 275 zones,
`2023-01-23 19:00` → `2023-02-28 23:00`). Lower MAE/RMSE is better; higher R² is better.

| model | MAE (kWh) | RMSE (kWh) | R² | vs baseline |
| --- | --- | --- | --- | --- |
| Persistence (`lag_1`) | 46.9833 | 265.9339 | 0.8893 | reference |
| Linear Regression | 43.2990 | 174.0560 | 0.9526 | beats baseline |
| **Random Forest** | **33.3272** | **151.7772** | **0.9639** | beats baseline |

A model is only reported as beating the baseline when its MAE, RMSE **and** R² are
all at least as good on the identical test rows. See `reports/model_comparison.md`
and `reports/model_metrics.json`.

### Not yet implemented

* Visualization utilities (`src/visualization/` is empty; the app uses built-in
  Streamlit charts).
* Weather features, station/zone encodings, rolling aggregates.
* Shared configuration (`config/`) and exploratory notebooks (`notebooks/`).
* A validation split / hyperparameter tuning (Random Forest settings are fixed).

## Workflow status

1. **Data acquisition** — done. UrbanEV raw data is present in `data/raw/`
   (not committed to version control).
2. **Data understanding** — partial. A read-only validation report exists at
   `reports/dataset_validation.txt`; exploratory notebooks are not yet written.
3. **Preprocessing** — done. `src/preprocessing/prepare_data.py` reshapes the wide
   time-by-zone matrix into a long table, adds calendar features, and produces a
   chronological per-zone train/test split.
4. **Feature engineering** — done (lags only). `src/features/demand_lags.py` adds
   `lag_1` and `lag_24` within each zone. Weather, station encodings, and rolling
   aggregates are not yet implemented.
5. **Modeling** — done. `src/models/train_models.py` fits Linear Regression and
   Random Forest on the training rows only, with a fixed random seed and a runtime
   guard against leaky predictor columns.
6. **Evaluation** — done. MAE, RMSE and R² for the baseline and both models are
   written to `reports/`, with an explicit per-model verdict against the baseline.
7. **Visualization** — partial. The Streamlit app charts model comparison and
   saved actual-vs-predicted series; `src/visualization/` is not yet populated.
8. **Application** — done. `app/streamlit_app.py` provides manual prediction,
   model comparison, saved-prediction charts and CSV downloads.
9. **Testing** — in progress. Tests live in `tests/`; see the testing section
   below for how they are run.

## Directory layout

```
data/raw/          original, immutable UrbanEV dataset (never modified)
data/interim/      intermediate data between pipeline stages (empty)
data/processed/    cleaned, model-ready data and generated predictions
notebooks/         exploratory notebooks (empty)
models/            persisted trained model artifacts (`linear_regression.joblib`,
                   `random_forest.joblib`)
reports/           validation and evaluation reports (dataset validation, baseline
                   and model metrics, markdown summaries)
results/           generated tables and predictions
  tables/          `model_comparison.csv`
  predictions/     per-model test predictions and the combined
                   baseline + models prediction file
  figures/         generated figures (empty)
src/               project source code
  data/            dataset validation utilities
  preprocessing/   wide-to-long reshaping, calendar features, chronological split
  features/        leakage-safe lag features
  models/          Linear Regression / Random Forest training and evaluation
  evaluation/      metrics and the persistence baseline
  visualization/   plotting utilities (no modules yet)
tests/             automated tests and manual verification scripts
  manual/          hand-run verification scripts (not collected by pytest)
app/               Streamlit dashboard (`streamlit_app.py`)
config/            shared configuration (empty)
```

Output files are written under `data/processed/` and `reports/` and are not
committed to version control (see `.gitignore`).

## Getting started

```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

Each stage is a runnable module and must be run **in this order** (each consumes
the previous stage's output):

```
python -m src.data.validate_dataset        # read-only dataset validation
python -m src.preprocessing.prepare_data   # -> data/processed/*
python -m src.features.demand_lags         # -> data/processed/features_*.csv
python -m src.evaluation.baseline_persistence  # -> baseline predictions + reports/
python -m src.models.train_models          # -> models/, reports/, results/
```

The model stage reads `reports/baseline_metrics.json`, so the baseline must be run
first. Pass `--no-write` to any stage for a dry run that writes nothing.

## Running the app

After the models are trained:

```
streamlit run app/streamlit_app.py
```

The app loads the saved `.joblib` models from `models/` and the comparison table
and predictions from `results/`. It offers a manual prediction form (calendar
features, `lag_1`, `lag_24`), a model-comparison table with best-metric cards, and
an actual-vs-predicted chart built from the saved test predictions.

## Verification scripts

Hand-run scripts in `tests/manual/` check the saved artifacts (executed directly,
not via pytest):

```
python tests/manual/verify_preprocessing.py   # preprocessing outputs
python tests/manual/verify_models.py          # saved models / predictions, read-only
```

## Tests

Tests live in `tests/` and are run with `pytest` (listed in `requirements.txt`):

```
pytest
```

`pytest.ini` sets `testpaths = tests` and puts the project root on the import
path, so no package installation is required. Note that `pytest` must be
installed first via `pip install -r requirements.txt`.

Some test modules also run without pytest, each with a self-contained runner that
uses in-memory fixtures (they train only on synthetic data and write no files):

```
python tests/test_demand_lags.py
python tests/test_train_models.py
```

`tests/test_validate_dataset.py` and `tests/test_prepare_data.py` import `pytest`
directly, so they require the full `pip install -r requirements.txt`. If that
dependency is not installed, run them through the same interpreter that has it
(e.g. `python -m pytest`), because running them as plain scripts will fail on the
import rather than run the checks.
