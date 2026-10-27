# Model comparison — one-hour-ahead EV charging demand

All numbers are on the **same chronological test rows** used by the
persistence baseline. Lower MAE/RMSE is better; higher R² is better.

## Test-set comparison

| model | MAE (kWh) | RMSE (kWh) | R² | vs baseline |
| --- | --- | --- | --- | --- |
| Persistence (lag_1) | 46.983300 | 265.933875 | 0.889306 | reference |
| Linear Regression | 43.299037 | 174.055987 | 0.952581 | beats baseline |
| Random Forest | 33.327187 | 151.777197 | 0.963943 | beats baseline |

## Deltas vs baseline (model − baseline)

| model | ΔMAE | ΔRMSE | ΔR² |
| --- | --- | --- | --- |
| Linear Regression | -3.684263 | -91.877887 | +0.063275 |
| Random Forest | -13.656113 | -114.156677 | +0.074637 |

## Configuration

* Predictors: `hour, day_of_week, month, lag_1, lag_24`
* Target: `demand_kwh`
* Excluded (leaky/identifier): `demand_kwh, time, zone_id, split, predicted_kwh, residual_kwh`
* Random seed: `42` (Random Forest; Linear Regression is deterministic)
* Fitted on: 949025 training rows (275 zones)
* Evaluated on: 238975 test rows (275 zones)

## Train metrics (diagnostic only — not evidence of generalisation)

| model | MAE (kWh) | RMSE (kWh) | R² |
| --- | --- | --- | --- |
| Linear Regression | 52.811077 | 202.672748 | 0.946188 |
| Random Forest | 35.965106 | 158.489730 | 0.967093 |

## Artifacts

* Linear Regression: model `C:\Users\Lenovo\Desktop\EV_charging_prediction\models\linear_regression.joblib`, predictions `C:\Users\Lenovo\Desktop\EV_charging_prediction\results\predictions\linear_regression_test_predictions.csv`
* Random Forest: model `C:\Users\Lenovo\Desktop\EV_charging_prediction\models\random_forest.joblib`, predictions `C:\Users\Lenovo\Desktop\EV_charging_prediction\results\predictions\random_forest_test_predictions.csv`
* metrics: `C:\Users\Lenovo\Desktop\EV_charging_prediction\reports\model_metrics.json`
* table:   `C:\Users\Lenovo\Desktop\EV_charging_prediction\results\tables\model_comparison.csv`
* combined predictions: `C:\Users\Lenovo\Desktop\EV_charging_prediction\results\predictions\model_test_predictions.csv`
