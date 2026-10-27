"""Independent verification of the saved model artifacts (read-only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
PRED = Path("results/predictions")


def load(p):
    return pd.read_csv(p, dtype={"zone_id": str}, parse_dates=["time"],
                       float_precision="round_trip")


ft = load(PROC / "features_test.csv")
base = load(PROC / "baseline_test_predictions.csv")
lr = load(PRED / "linear_regression_test_predictions.csv")
rf = load(PRED / "random_forest_test_predictions.csv")
M = json.load(open("reports/model_metrics.json"))

y = ft["demand_kwh"].to_numpy()

print("rows: lr=%d rf=%d base=%d features=%d" % (len(lr), len(rf), len(base), len(ft)))

aligned = (
    (lr["time"].to_numpy() == ft["time"].to_numpy()).all()
    and (lr["zone_id"].to_numpy() == ft["zone_id"].to_numpy()).all()
    and (rf["time"].to_numpy() == ft["time"].to_numpy()).all()
    and (rf["zone_id"].to_numpy() == ft["zone_id"].to_numpy()).all()
    and (base["time"].to_numpy() == ft["time"].to_numpy()).all()
)
print("all prediction files align to features_test order (time, zone):", bool(aligned))

same_y = (
    (lr["demand_kwh"].to_numpy() == y).all()
    and (rf["demand_kwh"].to_numpy() == y).all()
    and (base["demand_kwh"].to_numpy() == y).all()
)
print("y_true identical across all three prediction files:", bool(same_y))


def metrics(yp):
    e = y - yp
    return (
        float(np.mean(np.abs(e))),
        float(np.sqrt(np.mean(e ** 2))),
        1 - float(np.sum(e ** 2)) / float(np.sum((y - np.mean(y)) ** 2)),
    )


for name, yp, key in (
    ("LinearRegression", lr["predicted_kwh"].to_numpy(), "linear_regression"),
    ("RandomForest", rf["predicted_kwh"].to_numpy(), "random_forest"),
    ("Persistence", base["predicted_kwh"].to_numpy(), None),
):
    a, b, c = metrics(yp)
    print("%-16s mae=%.10f rmse=%.10f r2=%.10f" % (name, a, b, c))
    t = M["models"][key]["test"] if key else M["baseline_test"]
    print("   matches JSON exactly: mae=%s rmse=%s r2=%s"
          % (t["mae"] == a, t["rmse"] == b, t["r2"] == c))

print("predictions finite:", bool(np.isfinite(lr["predicted_kwh"]).all()
                                  and np.isfinite(rf["predicted_kwh"]).all()))
print("no NaN predictions:", bool(not lr["predicted_kwh"].isna().any()
                                  and not rf["predicted_kwh"].isna().any()))
print("beats-baseline flags:",
      {k: M["models"][k]["comparison"]["beats_baseline_on_all_three"]
       for k in M["models"]})
