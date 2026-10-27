# Persistence baseline — one-hour-ahead EV charging demand

## What was evaluated

The persistence rule predicts demand at time `t` with the demand observed at
`t-1`:

    y_pred = lag_1

No model was fitted. This is a reference point only: a later model must beat
these numbers **on the same test rows** to justify its added complexity.

## Test set (headline result)

| metric | value |
| --- | --- |
| MAE  (kWh) | 46.983300 |
| RMSE (kWh) | 265.933875 |
| R² | 0.889306 |
| rows | 238975 |
| zones | 275 |
| window | 2023-01-23 19:00:00 → 2023-02-28 23:00:00 |

## Training set (diagnostic only — NOT evidence of generalisation)

| metric | value |
| --- | --- |
| MAE  (kWh) | 57.950237 |
| RMSE (kWh) | 309.914655 |
| R² | 0.874174 |
| rows | 949025 |
| zones | 275 |
| window | 2022-09-02 00:00:00 → 2023-01-23 18:00:00 |

The train numbers describe how closely persistence tracks the series it is
being scored on. Because the split is chronological and the test window lies
strictly after the train window for every zone, the train row-count and window
are reported purely so the split is auditable.

## Leakage checks

* The prediction is an exact copy of `lag_1`, which the feature builder produced
  with a within-zone `shift(1)` on chronologically ordered history — a value
  genuinely known at `t-1`.
* `demand_kwh` is used **only** as the label in the metric computation; it never
  enters the prediction. Asserted at runtime for both splits.
* The existing chronological split is reused unchanged — no reshuffling, no
  refitting, and the test window is never used to build the forecast.
* No row was dropped or imputed at this stage: the feature files already contain
  no missing values (warm-up rows were dropped upstream).

## Artifacts

* predictions: `C:\Users\Lenovo\Desktop\EV_charging_prediction\data\processed\baseline_test_predictions.csv`
* metrics:     `C:\Users\Lenovo\Desktop\EV_charging_prediction\reports\baseline_metrics.json`
* report:      `C:\Users\Lenovo\Desktop\EV_charging_prediction\reports\baseline_persistence.md`
