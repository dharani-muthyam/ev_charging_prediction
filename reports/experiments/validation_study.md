# Time-based validation study (experiment layer)

The existing reference models and reports were **not** modified by this study.

## Design

* Search-train rows: 759275 (2022-09-02 00:00:00 → 2022-12-26 00:00:00)
* Validation rows: 189750 (2022-12-26 01:00:00 → 2023-01-23 18:00:00)
  — most recent 20% of each zone's reference-training rows
* Final test rows: 238975 (2023-01-23 19:00:00 → 2023-02-28 23:00:00)
* Selection rule: lowest validation RMSE, fixed **before** the test set was scored
* Random seed: 42

## Validation comparison

| candidate | MAE (kWh) | RMSE (kWh) | R² | fit s |
| --- | --- | --- | --- | --- |
| Random Forest (reference params + static zone features) | 34.5738 | 154.3146 | 0.9597 | 595.5 |
| Random Forest (reference: 100 trees, depth 20, leaf 20) | 35.4257 | 157.9501 | 0.9578 | 600.6 |
| HistGradientBoosting (400 iters, lr 0.05, leaf 100 + zone features) | 35.3258 | 160.2571 | 0.9565 | 34.2 |
| Random Forest (150 trees, depth 16, leaf 50) | 36.0479 | 161.6362 | 0.9558 | 556.8 |
| Random Forest (150 trees, depth 24, leaf 100) | 36.7140 | 164.3540 | 0.9543 | 2020.7 |
| HistGradientBoosting (400 iters, lr 0.05, leaf 100) | 38.7286 | 180.5856 | 0.9448 | 15.6 |

## Final test result (single candidate, scored once)

**Selected on validation:** Random Forest (reference params + static zone features)

| set | MAE (kWh) | RMSE (kWh) | R² |
| --- | --- | --- | --- |
| validation | 34.5738 | 154.3146 | 0.9597 |
| final test | 32.7890 | 151.1008 | 0.9643 |

## Versus the reference results (recorded, unmodified)

| model | MAE (kWh) | RMSE (kWh) | R² | same test rows |
| --- | --- | --- | --- | --- |
| reference random_forest | 33.3272 | 151.7772 | 0.9639 | yes |
| selected candidate | 32.7890 | 151.1008 | 0.9643 | True |

Beats the reference on **all three** test metrics: **YES** (ΔMAE -0.5382, ΔRMSE -0.6763, ΔR² +0.0003).

> validation ranking does not guarantee a test-set improvement; the verdict above is based on the measured final-test numbers only.
