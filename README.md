# EV Charging Demand Prediction Using Machine Learning

A reproducible machine-learning pipeline that forecasts **one-hour-ahead EV charging
demand** — the energy a given administrative zone will deliver in the next hour —
from the zone's own recent history (previous hour and same hour on the previous day)
plus calendar features. Trained models are served through an interactive
[Streamlit](https://streamlit.io/) dashboard.

The project turns a raw UrbanEV dataset into long-form, model-ready tables, builds
leakage-safe lag features on a chronological per-zone split, evaluates a model-free
persistence baseline, trains and compares two supervised regressors on **identical
test rows**, and writes every metric and artifact to disk so each experiment can be
audited and reproduced.

> **Status:** college/portfolio project. The reference pipeline, figures and dashboard
> are implemented and verified on the recorded test set. This is **not** a
> production-ready forecasting service — see [Limitations](#15-limitations).

---

## Table of contents

1. [Problem statement and motivation](#1-problem-statement-and-motivation)
2. [Project objectives](#2-project-objectives)
3. [Key features currently implemented](#3-key-features-currently-implemented)
4. [Technology stack](#4-technology-stack)
5. [Machine-learning workflow](#5-machine-learning-workflow)
6. [Models implemented](#6-models-implemented)
7. [Model performance](#7-model-performance)
8. [Dataset](#8-dataset)
9. [Project directory structure](#9-project-directory-structure)
10. [Installation (Windows + VS Code)](#10-installation-windows--vs-code)
11. [Running the Streamlit dashboard](#11-running-the-streamlit-dashboard)
12. [Running automated tests](#12-running-automated-tests)
13. [Reproducing training and validation](#13-reproducing-training-and-validation)
14. [Screenshots](#14-screenshots)
15. [Limitations](#15-limitations)
16. [Author](#16-author)

---

## 1. Problem statement and motivation

Urban EV charging demand is uneven: a handful of dense zones account for a large share
of total energy delivered, and demand swings strongly by hour of day and day of week.
Grid operators, charge-point operators and planners need a short-horizon estimate of
how much energy each zone will deliver next hour.

A trivial "next hour looks like this hour" rule (persistence) is a strong contender on
smooth hourly load data. Any forecasting model must therefore be shown to beat that
rule **on the same rows** before its added complexity is justified. This project
builds exactly that comparison: a baseline, two trained regressors, and an honest,
fixed evaluation protocol.

## 2. Project objectives

- Build a reproducible, auditable pipeline from raw charging records to trained models.
- Define an explicit, leakage-safe prediction task: `demand_kwh[z, t]` from information
  available at `t-1`.
- Establish a model-free **persistence baseline** as the reference to beat.
- Train and compare supervised regressors on the identical chronological test rows.
- Report MAE, RMSE and R² and state a per-model verdict against the baseline rather
  than an unfounded "improvement" claim.
- Expose the saved artifacts through a read-only dashboard for inspection and manual
  what-if prediction.

## 3. Key features currently implemented

Stages run in the order shown; each consumes the previous stage's output.

| Stage | Module | Output |
| --- | --- | --- |
| Dataset validation (read-only) | `src/data/validate_dataset.py` | `reports/dataset_validation.txt` |
| Preprocessing (wide → long, calendar features, chronological split) | `src/preprocessing/prepare_data.py` | `data/processed/demand_long.csv`, `train.csv`, `test.csv`, `split_metadata.csv`, `split_summary.json` |
| Lag features (`lag_1`, `lag_24`, leakage-safe, per zone) | `src/features/demand_lags.py` | `data/processed/features_lag.csv`, `features_train.csv`, `features_test.csv`, `features_summary.json` |
| Static zone attributes (leak-free metadata join) | `src/features/zone_features.py` | `data/processed/features_train_zone.csv`, `features_test_zone.csv`, `zone_features_summary.json` |
| Persistence baseline (`y_pred = lag_1`) | `src/evaluation/baseline_persistence.py` | `data/processed/baseline_test_predictions.csv`, `reports/baseline_metrics.json`, `reports/baseline_persistence.md` |
| Model training and evaluation (Linear Regression, Random Forest) | `src/models/train_models.py` | `models/*.joblib`, `reports/model_metrics.json`, `reports/model_comparison.md`, `results/tables/model_comparison.csv`, `results/predictions/*` |
| Time-based validation study (experiment layer) | `src/models/validation_study.py` | `reports/experiments/validation_study.{json,md}`, `results/experiments/validation_study.csv` |
| Evaluation figures | `src/visualization/plots.py` | `results/figures/fig_*.png` (5 figures) |
| Interactive dashboard | `app/streamlit_app.py` | Streamlit UI (see §11) |
| Automated tests | `tests/` | `pytest` suite (79 tests) |

Additional safeguards are **enforced in code**, not merely documented:

- `FORBIDDEN_PREDICTORS` in `src/models/train_models.py` is checked at runtime by
  `assert_predictors_are_safe`, so a leaky or identifier column cannot be used as a
  predictor.
- Lags are built with a within-zone `shift`, so no lag value ever crosses a zone
  boundary; rows without a complete lag set are dropped, never imputed.
- In the reference pipeline the test period is never used for fitting or for model
  selection.

## 4. Technology stack

| Layer | Tools |
| --- | --- |
| Language | Python 3 (developed and tested with CPython 3.13) |
| Data handling | `pandas` ≥ 2.2, `numpy` ≥ 1.26 |
| Modelling | `scikit-learn` ≥ 1.4 (`LinearRegression`, `RandomForestRegressor`, and `HistGradientBoostingRegressor` in the validation study) |
| Model persistence | `joblib` |
| Visualisation | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` ≥ 1.40 |
| Testing | `pytest` ≥ 7 |
| Versioning aids | Git LFS for `*.joblib`, `*.zip`, `*.csv` (`.gitattributes`) |

Exact version ranges are pinned in [`requirements.txt`](requirements.txt).

## 5. Machine-learning workflow

### 5.1 Data preparation

1. **Validation (read-only).** `src/data/validate_dataset.py` checks each hourly
   zone-level CSV for shape, duplicate timestamps/zone columns, missing cells,
   non-numeric and negative values, statistical ranges and hourly continuity. It never
   writes back, clips or transforms the raw files.
2. **Wide → long reshape.** `src/preprocessing/prepare_data.py` reads
   `charge_1hour/volume.csv` (4,344 hourly timestamps × 275 zones), stacks it into
   1,194,600 `(time, zone_id, demand_kwh)` rows, and adds `hour`, `day_of_week` and
   `month` columns. Demand values are carried through **unchanged** — no imputation,
   clipping, scaling or synthetic records.
3. **Exclusion rationale.** `occupancy.csv` and `duration.csv` are excluded because
   they are _same-hour_ observations of the interval being predicted (direct target
   leakage). The measurement and decision are documented in
   [`docs/data_quality.md`](docs/data_quality.md).

### 5.2 Feature engineering

Two families of predictors are available:

| Feature set | Columns | Built by |
| --- | --- | --- |
| Base          | `hour`, `day_of_week`, `month`, `lag_1`, `lag_24`                                                                                            | `prepare_data.py` + `demand_lags.py`         |
| Zone-extended | base + `zone_charge_count`, `zone_area_km2`, `zone_perimeter_km`, `zone_piles_per_km2`, `zone_n_stations`, `zone_longitude`, `zone_latitude` | `zone_features.py` (many-to-one static join) |

- `lag_1` = demand at `t-1`; `lag_24` = demand at `t-24` (same hour, previous day).
- Both are shifted backwards within each zone, so every predictor is known at or before
  `t-1`.
- The first 24 rows of each zone have incomplete lags and are dropped (6,600 rows in
  total), leaving 1,188,000 feature rows.
- Zone attributes are administrative constants (pile counts, area, centroid), known at
  forecast time and therefore leak-free.

### 5.3 Train/test split

A **chronological per-zone 80/20 split** — the earliest 80 % of each zone's timestamps
go to train, the latest 20 % to test. Timestamps are never shuffled.

| Split | Rows | Zones | Window |
| --- | --- | --- | --- |
| Train (after lag warm-up drop) | 949,025 | 275   | 2022-09-02 00:00 → 2023-01-23 18:00 |
| Test                           | 238,975 | 275   | 2023-01-23 19:00 → 2023-02-28 23:00 |

### 5.4 Model training

`src/models/train_models.py` fits each estimator on the training rows only, with a
fixed random seed (`42`) for the Random Forest; Linear Regression is deterministic. The
fitted estimator, the predictor list and the seed are persisted together. No shuffling
or re-splitting occurs.

### 5.5 Model evaluation

Predictions are made on the test rows and scored with MAE, RMSE and R². The persisted
baseline metrics are read from `reports/baseline_metrics.json` rather than recomputed,
so the comparison is against the recorded numbers on the identical rows. A model is
only reported as beating the baseline when **MAE and RMSE are lower and R² is higher**
on the same rows (`compare_to_baseline`).

### 5.6 Prediction through the dashboard

`app/streamlit_app.py` loads the saved `.joblib` estimators and asks for the five base
predictors, then calls `model.predict` on a single-row frame. The app checks the
model's `feature_names_in_` before predicting and reports an explicit error instead of
silently substituting a missing feature. It never retrains anything.

## 6. Models implemented

| Model | Where | Configuration | Notes |
| --- | --- | --- | --- |
| Persistence baseline                   | `src/evaluation/baseline_persistence.py` | `y_pred = lag_1`                                                                          | No model is fitted; reference every model must beat.                                 |
| Linear Regression                      | `src/models/train_models.py`             | `sklearn.linear_model.LinearRegression()` (OLS)                                           | Deterministic, interpretable, linear only.                                           |
| Random Forest                          | `src/models/train_models.py`             | `n_estimators=100`, `max_depth=20`, `min_samples_leaf=20`, `random_state=42`, `n_jobs=-1` | Depth/leaf bounds keep the artifact storable while retaining accuracy.               |
| HistGradientBoosting (experiment only) | `src/models/validation_study.py`         | `max_iter=400`, `learning_rate=0.05`, `max_leaf_nodes=100`                                | Evaluated on validation only; **not** the reported model and not saved to `models/`. |

The reference pipeline (`train_models.py`) uses the **base feature set**. Static zone
features are used only in the separate validation-study experiment.

## 7. Model performance

All numbers are on the **same chronological test rows** (238,975 rows, 275 zones,
2023-01-23 19:00 → 2023-02-28 23:00). Lower MAE/RMSE is better; higher R² is better.
Source: [`reports/model_comparison.md`](reports/model_comparison.md) and
[`reports/model_metrics.json`](reports/model_metrics.json).

| Model | MAE (kWh) | RMSE (kWh) | R² | vs baseline |
| --- | --- | --- | --- | --- |
| Persistence (`lag_1`) | 46.9833     | 265.9339     | 0.8893     | reference             |
| Linear Regression     | 43.2990     | 174.0560     | 0.9526     | beats baseline        |
| **Random Forest**     | **33.3272** | **151.7772** | **0.9639** | beats baseline (best) |

Deltas versus the baseline (model − baseline):

| Model | ΔMAE | ΔRMSE | ΔR² |
| --- | --- | --- | --- |
| Linear Regression | −3.6843  | −91.8779  | +0.0633 |
| Random Forest     | −13.6561 | −114.1567 | +0.0746 |

Train-set metrics (diagnostic only — **not** evidence of generalisation):

| Model | MAE (kWh) | RMSE (kWh) | R² |
| --- | --- | --- | --- |
| Linear Regression | 52.8111   | 202.6727   | 0.9462 |
| Random Forest     | 35.9651   | 158.4897   | 0.9671 |

### Validation study (separate experiment, reference reports untouched)

[`reports/experiments/validation_study.md`](reports/experiments/validation_study.md)
compares six candidate configurations on a validation fold (189,750 rows — the most
recent 20 % of each zone's reference-training rows) and scores the winner **once** on
the final test rows. Selection rule: lowest validation RMSE, fixed before the test set
was scored.

| Candidate | Val MAE | Val RMSE | Val R² |
| --- | --- | --- | --- |
| Random Forest (reference params + static zone features)             | 34.5738 | 154.3146 | 0.9597 |
| Random Forest (reference: 100 trees, depth 20, leaf 20)             | 35.4257 | 157.9501 | 0.9578 |
| HistGradientBoosting (400 iters, lr 0.05, leaf 100 + zone features) | 35.3258 | 160.2571 | 0.9565 |
| Random Forest (150 trees, depth 16, leaf 50)                        | 36.0479 | 161.6362 | 0.9558 |
| Random Forest (150 trees, depth 24, leaf 100)                       | 36.7140 | 164.3540 | 0.9543 |
| HistGradientBoosting (400 iters, lr 0.05, leaf 100)                 | 38.7286 | 180.5856 | 0.9448 |

Selected on validation: **Random Forest (reference params + static zone features)**.
Final test: MAE 32.7890 kWh, RMSE 151.1008 kWh, R² 0.9643 — a small improvement over the
reference Random Forest (ΔMAE −0.5382, ΔRMSE −0.6763, ΔR² +0.0003). This candidate is
recorded as an experiment only; it is **not** saved to `models/`, and the reference
metrics above remain the reported results.

## 8. Dataset

**UrbanEV** — an open benchmark dataset for urban EV charging demand prediction,
collected from public charging stations in Shenzhen, China, covering
**1 September 2022 – 28 February 2023**. The bundled documentation
([`data/raw/README.md`](data/raw/README.md)) describes the archive and links to the
public project repository (https://github.com/IntelligentSystemsLab/UrbanEV).

Files used by this project, all from
`data/raw/UrbanEVDataset/20220901-20230228_zone-cleaned-aggregated/`:

| File | Use in this project |
| --- | --- |
| `charge_1hour/volume.csv`                    | Target source — aggregated charging volume per zone per hour (kWh). Wide format: first column `time`, columns 2–276 are zone IDs (TAZID). |
| `zone-information.csv`                       | Static zone attributes (pile count, area, perimeter, centroid) for the validation-study zone features.                                    |
| `station_information.csv`                    | Per-station attributes, aggregated to zone level for `zone_n_stations`.                                                                   |
| `charge_1hour/occupancy.csv`, `duration.csv` | Validated but **not** used as predictors (same-hour leakage — see `docs/data_quality.md`).                                                |
| `charge_1hour/volume-11kw.csv`               | Validated but not used.                                                                                                                   |

Verified structure of `volume.csv` (`reports/dataset_validation.txt`): 4,344 rows ×
275 zone columns, hourly and gap-free, 0 duplicate timestamps, 0 missing cells, 0
negative cells; values range 0 – 16,732.5 kWh (mean 261.46 kWh).

The raw dataset is not committed (see §9). Place it under `data/raw/UrbanEVDataset/`
before running the pipeline. Dataset licensing and terms of use are governed by the
original UrbanEV project; consult that repository.

## 9. Project directory structure

```
EV_charging_prediction/
├── app/
│   └── streamlit_app.py         Streamlit dashboard (read-only over saved artifacts)
├── config/                      shared configuration (empty)
├── data/
│   ├── raw/                     original UrbanEV dataset (ignored by Git)
│   ├── interim/                 intermediate data (empty)
│   └── processed/               generated model-ready tables and predictions
├── docs/
│   └── data_quality.md          occupancy/duration audit and exclusion rationale
├── models/
│   ├── linear_regression.joblib
│   └── random_forest.joblib
├── notebooks/                   exploratory notebooks (empty)
├── reports/
│   ├── dataset_validation.txt
│   ├── baseline_metrics.json
│   ├── baseline_persistence.md
│   ├── model_metrics.json
│   ├── model_comparison.md
│   └── experiments/             validation_study.json / .md
├── results/
│   ├── figures/                 fig_*.png evaluation figures
│   ├── predictions/             per-model and combined test predictions
│   ├── tables/                  model_comparison.csv
│   └── experiments/             validation_study.csv
├── src/
│   ├── data/validate_dataset.py
│   ├── preprocessing/prepare_data.py
│   ├── features/demand_lags.py
│   ├── features/zone_features.py
│   ├── evaluation/baseline_persistence.py
│   ├── models/train_models.py
│   ├── models/validation_study.py
│   └── visualization/plots.py
├── tests/
│   ├── test_*.py                pytest suite
│   └── manual/                  hand-run verification scripts
├── pytest.ini
├── requirements.txt
└── README.md
```

### Generated artifacts in Git

Some files produced by the pipeline are tracked so the results are auditable on a
fresh clone; the large per-row tables are ignored and must be regenerated. See
`.gitignore` (its `!` exception rules) and `.gitattributes` (LFS for `*.csv`,
`*.joblib`, `*.zip`).

**Tracked in Git**

| Path | Kind |
| --- | --- |
| `data/processed/split_summary.json`, `split_metadata.csv` | split manifest and per-zone split metadata |
| `data/processed/features_summary.json`, `zone_features_summary.json` | feature-build summaries |
| `data/processed/baseline_test_predictions.csv` | baseline test predictions |
| `results/tables/model_comparison.csv` | test-set MAE/RMSE/R² comparison |
| `models/linear_regression.joblib`, `models/random_forest.joblib` | trained models (tracked via Git LFS) |
| `reports/` (all contents) | validation, baseline, model and validation-study reports |
| `results/tables/validation_study.csv`, `reports/experiments/*` | validation-study outputs |

**Ignored (generated, not committed)**

| Path | Why |
| --- | --- |
| `data/raw/*` | the large raw UrbanEV dataset |
| `data/processed/demand_long.csv`, `train.csv`, `test.csv` | large intermediate tables |
| `data/processed/features_lag.csv`, `features_train.csv`, `features_test.csv` | large feature tables |
| `data/processed/features_train_zone.csv`, `features_test_zone.csv` | large zone-enriched tables |
| `results/predictions/*` | per-row prediction files (regenerable) |
| `results/figures/*` | generated PNG figures (regenerable) |
| `results/experiments/validation_study.csv` | generated experiment table |

Note: `results/tables/validation_study.csv` matches the `.gitignore` exception for that
filename, but the validation study writes its table to
`results/experiments/validation_study.csv`. Only `results/tables/model_comparison.csv`
and the `.gitkeep` placeholders are actually tracked under `results/tables/`.

## 10. Installation (Windows + VS Code)

**Prerequisites:** Python 3.10+ (developed with 3.13), Git and, for opening the
project, VS Code.

1. Clone or download the repository and open the folder in VS Code
   (`File → Open Folder…`).

2. Create and activate a virtual environment — in the VS Code integrated terminal
   (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks the activation script, use
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for that session only,
   or activate from the Command Prompt instead: `.venv\Scripts\activate.bat`.

3. Select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** →
   `.venv\Scripts\python.exe`.

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Place the UrbanEV data so that this path exists:

   ```
   data/raw/UrbanEVDataset/20220901-20230228_zone-cleaned-aggregated/charge_1hour/volume.csv
   ```

All pipeline commands below are run from the **project root**. They work as
`python -m src.…` without installing the package.

## 11. Running the Streamlit dashboard

After the models have been trained (§13):

```powershell
streamlit run app/streamlit_app.py
```

The app opens a local browser page (the URL is printed in the terminal, typically
`http://localhost:8501`). It is read-only over the saved artifacts and contains:

1. **Predict charging demand** — choose a trained model, enter `lag_1`, `lag_24`,
   hour, day of week and month, and get a predicted kWh value with a CSV download.
2. **Model comparison** — table of MAE/RMSE/R² with best-metric cards, an error bar
   chart, and a CSV download.
3. **Test-set results** — actual vs. predicted hourly series (per zone or all zones
   averaged) and an error-analysis tab (MAE by model, worst zones, error by demand
   level) from the saved `model_test_predictions.csv`.
4. **Validation study** — optional; shown only if
   `reports/experiments/validation_study.json` exists.
5. **Generated figures** — optional; shown only if images exist in
   `results/figures/`.
6. **Models and limitations** — short model descriptions and known caveats.

If an artifact is missing, the corresponding section reports that and is skipped.

## 12. Running automated tests

`pytest.ini` sets `testpaths = tests` and puts the project root on `sys.path`, so the
pipeline modules can be imported without installing the package. `pytest` itself is
installed by `pip install -r requirements.txt` (see §10).

Run the whole suite from the project root:

```powershell
python -m pytest -q
```

The suite (79 tests collected) covers dataset validation, wide-to-long reshaping and
the chronological split, lag construction and zone isolation, static zone-feature
joins, predictor leakage guards, model training/evaluation on synthetic data, and the
validation split. Tests use in-memory or synthetic fixtures and do not write project
files.

Separately, two hand-run scripts verify the **saved artifacts**. They are executed
directly rather than collected by `pytest`, because they run at import time and call
`sys.exit`:

```powershell
python tests/manual/verify_preprocessing.py   # preprocessing outputs vs raw data
python tests/manual/verify_models.py          # saved models/predictions, read-only
```

## 13. Reproducing training and validation

Each stage is a runnable module and must be run **in this order** — each consumes the
previous stage's output:

```powershell
python -m src.data.validate_dataset            # read-only  -> reports/dataset_validation.txt
python -m src.preprocessing.prepare_data       # -> data/processed/*
python -m src.features.demand_lags             # -> data/processed/features_*.csv
python -m src.features.zone_features           # -> data/processed/features_*_zone.csv
python -m src.evaluation.baseline_persistence  # -> baseline predictions + reports/
python -m src.models.train_models              # -> models/, reports/, results/
python -m src.visualization.plots              # -> results/figures/fig_*.png
```

The model stage reads `reports/baseline_metrics.json`, so the baseline must run first.
Stages that produce files accept `--no-write` for a dry run that writes nothing.

The optional validation study (experiment layer) needs the zone-enriched tables from
`src.features.zone_features` and writes only to `reports/experiments/` and
`results/experiments/` — it never overwrites the reference models or reports:

```powershell
python -m src.models.validation_study              # full study
python -m src.models.validation_study --quick      # sampled smoke run
```

Runtime note: fitting the Random Forest on ~950k training rows takes minutes and the
saved artifact is large (approximately 160 MB). With `n_jobs=-1` the Random Forest is
reproducible to roughly 1e-12 rather than bit-for-bit; Linear Regression is exactly
reproducible.

## 14. Screenshots

<!-- Replace the placeholders below with screenshots of your running app. -->

| Dashboard | Prediction form |
| --- | --- |
| _Screenshot placeholder — main dashboard_ | _Screenshot placeholder — manual prediction result_ |

## 15. Limitations

- **Short horizon only.** The task is one hour ahead; there is no multi-step or
  day-ahead forecasting.
- **Limited feature set.** The reference models use two lags plus calendar features.
  Weather, holidays, events, rolling aggregates and spatial/adjacency information are
  not used even though the dataset provides some of them.
- **No per-zone model identity in the reference models.** `zone_id` is not a predictor,
  and static zone attributes appear only in the validation-study experiment.
- **Errors concentrate at peak load.** A single MAE understates uncertainty for the
  highest-demand zones and hours.
- **Validation-study candidate is not deployed.** The zone-feature Random Forest beats
  the reference by a small margin but is not persisted to `models/`.
- **Single chronological split.** No rolling-origin backtesting; train/test metrics are
  reported from one period.
- **Dashboard predicts one row at a time.** The manual form is a demonstration of the
  saved estimators, not a batch or per-zone service, and a regression can return a
  negative value (flagged as a statistical artefact, not physical).

Possible future improvements (none implemented yet): add weather and holiday features,
rolling/same-week lags, `zone_id` encodings or per-zone models, rolling-origin
cross-validation, and persisting and serving the best validation-study configuration.

## 16. Author

_EV Charging Demand Prediction Using Machine Learning_ is a B.Tech project. Author
details are not recorded in the repository — add your name, course and institution
here.
