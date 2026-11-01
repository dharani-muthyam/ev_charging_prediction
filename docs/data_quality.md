# Data quality investigation: `occupancy.csv` and `duration.csv`

This note records **what was measured** and **what decision follows**, for the two
raw zone-level files the original validation report flagged as suspicious. Raw
files were never modified, clipped, or repaired.

Files inspected (read-only):

```
data/raw/UrbanEVDataset/20220901-20230228_zone-cleaned-aggregated/charge_1hour/
    occupancy.csv
    duration.csv
    volume.csv          (the demand target, for cross-checking)
data/raw/UrbanEVDataset/20220901-20230228_zone-cleaned-aggregated/
    zone-information.csv
```

## 1. Schema and completeness

Both files have the same shape as `volume.csv`: **4,344 hourly timestamps × 275
zones** (`2022-09-01 00:00` → `2023-02-28 23:00`).

| check | occupancy.csv | duration.csv |
| --- | --- | --- |
| missing cells | 0 | 0 |
| duplicate timestamps | 0 | 0 |
| duplicate zone columns | 0 | 0 |
| hourly, gap-free | yes | yes |
| zone columns match `volume.csv` | yes | yes |
| zone set matches `zone-information.csv` (275 TAZIDs) | yes | yes |
| min / max / mean | 0 / 373 / 17.87 | 0 / 207.58 / 13.11 |

So both files are structurally clean and can be **joined one-to-one** to the
demand data on `(time, zone_id)`. The join is not the problem — timing is.

## 2. Occupancy: what the values mean

The dataset README documents `occupancy.csv` as the *"occupancy rate of charging
stations per time interval (Unit: %)"*. The audit flagged **10,980 cells above
100** (0.92 % of 1,194,600), maximum 373.

Three lines of evidence, all measured:

1. **The excess is concentrated, not random.** Only **31 of 275 zones** ever
   exceed 100; 10,980 of these cells belong largely to a handful of zones
   (top zone 1167 has 3,317 such cells, max 373).
2. **Those zones are the dense ones.** Zones that exceed 100 have a mean
   `charge_count` of **213.8** piles versus **68.9** for the rest. They hold
   **35.9 %** of all charging volume in the dataset.
3. **The values are internally consistent with the demand target.** Across all
   cells, `occupancy` correlates **0.967** with `volume` and **0.967** with
   `duration`; the hours with occupancy > 100 have a mean volume of 2,347 kWh
   versus 242 kWh for 0–100 cells. A percentage that were truly broken would not
   track the energy delivered this closely.

**Interpretation:** the 0–100 range is not an enforced bound in the published
file. The most plausible reading is that occupancy is a station-utilisation
measure (occupied pile-hours ÷ available pile-hours × 100) whose upper bound
depends on how piles and sessions are counted in a zone-hour, so values > 100
are *meaningful high-utilisation readings*, not obvious corruption. This is a
hypothesis consistent with the data, not a documented fact — the dataset README
gives no upper bound.

**Decision: none of these values was clipped, deleted, or rescaled.** Clipping
to 100 would fabricate an upper bound the dataset does not state and would
distort the very zones that carry the most demand.

## 3. Duration

`duration.csv` is the aggregated charging duration per zone per hour (unit:
hours). It has no missing values, no duplicates, and no values outside its own
documented domain. It correlates **0.995** with `volume`, which is expected
because the dataset computes volume from duration × pile power.

## 4. The decisive issue: both are *contemporaneous*

The modelling task is **one-hour-ahead**: predict `demand_kwh[z, t]` using
information available at `t-1`.

`occupancy[z, t]` and `duration[z, t]` are **observations of the same hour being
predicted**. Using them as predictors would be direct target leakage — the model
would be told part of the answer. A lagged version (`occupancy[z, t-1]`) would
be legitimate, but it duplicates information already carried by `lag_1`
(last hour's demand), which is itself 0.967-correlated with occupancy.

**Decision:**

* **Not used as predictors**, in either raw or lagged form. They are listed in
  `FORBIDDEN_PREDICTORS` in `src/models/train_models.py` so the guard rejects
  any attempt to add them, and the exclusion is recorded in
  `data/processed/zone_features_summary.json`.
* **Clean as time-series data** (0 missing, 0 duplicates), but *not* a valid
  feature source for this forecasting horizon.

## 5. Why zone metadata *is* used instead

`zone-information.csv` and `station_information.csv` contain **static**
attributes (pile count, area, perimeter, station count, centroid). A zone's pile
count does not change hour to hour and is known when a forecast is requested, so
those attributes are leak-free predictors — see
`src/features/zone_features.py` and `data/processed/zone_features_summary.json`.

## 6. Reproducing these checks

```powershell
python -m src.data.validate_dataset      # structural report -> reports/dataset_validation.txt
python -m src.features.zone_features     # join/safety report -> data/processed/zone_features_summary.json
```

The numbers in this note come from the raw files as shipped; re-running the two
commands above reproduces the structural and join findings.
