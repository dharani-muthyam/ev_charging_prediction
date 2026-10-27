# EV Charging Demand Prediction Using Machine Learning

A machine learning project that predicts electric vehicle (EV) charging demand —
the expected charging load at a given station or region over a future time window —
using historical charging session records and contextual data (time of day, day of
week, weather, station attributes, and traffic/usage patterns).

The goal is to produce a reproducible pipeline that turns raw session logs into a
validated forecasting model, with all results and metrics saved to disk so that
every experiment can be audited and compared.

## Project status

Pipeline through a validated persistence baseline. The UrbanEV raw dataset is in
place under `data/raw/`; validation, preprocessing, lag-feature generation, and a
model-free persistence baseline are implemented and verified. **No machine-learning
model has been trained yet** (no Linear Regression, Random Forest, or other
regressor), and the Streamlit interface in `app/` is **not implemented**.

### Implemented and verified

| Stage | Module | Output |
| --- | --- | --- |
| Dataset validation (read-only) | `src/data/validate_dataset.py` | `reports/dataset_validation.txt` |
| Preprocessing (wide → long, calendar features, chronological split) | `src/preprocessing/prepare_data.py` | `data/processed/demand_long.csv`, `train.csv`, `test.csv`, `split_metadata.csv`, `split_summary.json` |
| Lag features (`lag_1`, `lag_24`, leakage-safe, per zone) | `src/features/demand_lags.py` | `data/processed/features_lag.csv`, `features_train.csv`, `features_test.csv`, `features_summary.json` |
| Persistence baseline (`y_pred = lag_1`) | `src/evaluation/baseline_persistence.py` | `data/processed/baseline_test_predictions.csv`, `reports/baseline_metrics.json`, `reports/baseline_persistence.md` |

**Persistence baseline (test set, headline result):** MAE 46.9833 kWh,
RMSE 265.9339 kWh, R² 0.8893. These are a reference point only — any later model
must beat them on the same test rows. See `reports/baseline_persistence.md`.

### Not yet implemented

* Machine-learning models (`src/models/` is empty of modules).
* Visualization (`src/visualization/`).
* Streamlit interface (`app/`).
* Shared configuration (`config/`) and exploratory notebooks (`notebooks/`).

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
5. **Modeling** — not started. Only the model-free persistence baseline exists.
6. **Evaluation** — baseline metrics done (MAE, RMSE, R²) and written to
   `reports/`; comparison of trained models is not started.
7. **Visualization** — not started.
8. **Application** — not started.
9. **Testing** — in progress. Tests live in `tests/`; see the testing section
   below for how they are run.

## Directory layout

```
data/raw/          original, immutable UrbanEV dataset (never modified)
data/interim/      intermediate data between pipeline stages (empty)
data/processed/    cleaned, model-ready data and generated predictions
notebooks/         exploratory notebooks (empty)
models/            persisted trained model artifacts (empty)
reports/           validation and evaluation reports
results/           generated figures / tables / predictions (empty)
src/               project source code
  data/            dataset validation utilities
  preprocessing/   wide-to-long reshaping, calendar features, chronological split
  features/        leakage-safe lag features
  models/          model training and inference (no modules yet)
  evaluation/      metrics and the persistence baseline
  visualization/   plotting utilities (no modules yet)
tests/             automated tests and manual verification scripts
  manual/          hand-run verification scripts (not collected by pytest)
app/               Streamlit application (not implemented)
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

Each stage is a runnable module:

```
python -m src.data.validate_dataset        # read-only dataset validation
python -m src.preprocessing.prepare_data   # -> data/processed/*
python -m src.features.demand_lags         # -> data/processed/features_*.csv
python -m src.evaluation.baseline_persistence  # -> predictions + reports/
```

A hand-run verification script for the preprocessing outputs lives at
`tests/manual/verify_preprocessing.py` (it is executed directly, not via pytest):

```
python tests/manual/verify_preprocessing.py
```

## Tests

Tests live in `tests/` and are run with `pytest` (listed in `requirements.txt`):

```
pytest
```

`pytest.ini` sets `testpaths = tests` and puts the project root on the import
path, so no package installation is required. Note that `pytest` must be
installed first via `pip install -r requirements.txt`.

Some test modules also run without pytest: `tests/test_demand_lags.py` has a
self-contained runner (`python tests/test_demand_lags.py`).
