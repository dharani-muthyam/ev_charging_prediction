"""Independent verification of the preprocessing outputs.

Deliberately does NOT import src.preprocessing.prepare_data: the raw file is
re-parsed from scratch and compared against the on-disk processed CSVs, so this
is a genuine second opinion rather than a re-run of the same code path.

Read-only: nothing under data/ is modified.

Location: ``tests/manual/verify_preprocessing.py``. This script is run by hand
(not collected by pytest, because it executes at import time and calls
``sys.exit``). The project root is therefore two directories up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# tests/manual/verify_preprocessing.py -> tests/manual -> tests -> project root
ROOT = Path(__file__).resolve().parents[2]
RAW = (
    ROOT / "data" / "raw" / "UrbanEVDataset"
    / "20220901-20230228_zone-cleaned-aggregated" / "charge_1hour" / "volume.csv"
)
PROC = ROOT / "data" / "processed"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------- raw reload
raw = pd.read_csv(RAW, index_col=0, parse_dates=True, float_precision="round_trip")
raw.index.name = "time"
raw.columns = raw.columns.astype(str)
raw_long = (
    raw.rename_axis(index="time", columns="zone_id")
    .stack()
    .rename("demand_kwh")
    .reset_index()
)
raw_long["zone_id"] = raw_long["zone_id"].astype(str)
raw_long["time"] = pd.to_datetime(raw_long["time"])
raw_long["demand_kwh"] = raw_long["demand_kwh"].astype("float64")
raw_long = raw_long.sort_values(["zone_id", "time"], kind="mergesort").reset_index(drop=True)

check("raw wide shape == 4344 x 275", raw.shape == (4344, 275), str(raw.shape))
check("raw n_zones == 275", raw_long["zone_id"].nunique() == 275,
      str(raw_long["zone_id"].nunique()))


# ---------------------------------------------------------------- processed
def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"zone_id": str}, parse_dates=["time"],
                       float_precision="round_trip")


dl = load(PROC / "demand_long.csv")
tr = load(PROC / "train.csv")
te = load(PROC / "test.csv")
sm = pd.read_csv(PROC / "split_metadata.csv", dtype={"zone_id": str})
meta = sm  # split metadata read as text for zone_id

# ---- exact numeric preservation (bit-for-bit via tobytes)
exp = raw_long["demand_kwh"].to_numpy()
got = dl["demand_kwh"].to_numpy()
check("demand_long row count == 1194600", len(dl) == 1_194_600, str(len(dl)))
check("demand_long row order == raw sorted order",
      dl[["time", "zone_id"]].equals(raw_long[["time", "zone_id"]]))
check("demand_kwh bit-exact vs raw reload",
      exp.tobytes() == got.tobytes(),
      f"mismatches={(exp != got).sum()}")

# independent string-level check of the serialization hypothesis
raw_strs = pd.read_csv(RAW, index_col=0, dtype=str).stack().to_numpy()
proc_strs = pd.read_csv(PROC / "demand_long.csv", dtype=str)["demand_kwh"].to_numpy()
check("demand_long serialized strings match raw strings exactly",
      bool(np.array_equal(raw_strs, proc_strs)),
      f"string mismatches={int((raw_strs != proc_strs).sum())}")

# ---- dtypes / types
check("zone_id dtype is string/object in demand_long",
      dl["zone_id"].dtype == object or str(dl["zone_id"].dtype).startswith("str"))
check("zone_id all str instances", all(isinstance(v, str) for v in dl["zone_id"]))
check("demand_kwh dtype float64", dl["demand_kwh"].dtype == np.float64,
      str(dl["demand_kwh"].dtype))
check("time parsed as datetime64",
      pd.api.types.is_datetime64_any_dtype(dl["time"]))

# ---- sorting
z = dl["zone_id"].tolist()
check("zone_id textually sorted", z == sorted(z))
per_zone_mono = all(g["time"].is_monotonic_increasing
                    for _, g in dl.groupby("zone_id", sort=False))
check("time monotonic increasing within each zone", per_zone_mono)

# ---- missing values
na_total = int(dl[["time", "zone_id", "demand_kwh", "hour", "day_of_week", "month"]].isna().sum().sum())
check("no missing values in demand_long", na_total == 0, f"na={na_total}")

# ---- duplicates on (zone_id, time)
dup_keys = int(dl.duplicated(subset=["zone_id", "time"]).sum())
check("no duplicate (zone_id, time) keys in demand_long", dup_keys == 0, f"dups={dup_keys}")

# ---- calendar features correctness (spot + full)
ts = dl["time"]
check("hour == time.hour (all rows)", bool((dl["hour"].to_numpy() == ts.dt.hour.to_numpy()).all()))
check("day_of_week == time.dayofweek (all rows)",
      bool((dl["day_of_week"].to_numpy() == ts.dt.dayofweek.to_numpy()).all()))
check("month == time.month (all rows)",
      bool((dl["month"].to_numpy() == ts.dt.month.to_numpy()).all()))
check("calendar dtypes int16",
      str(dl["hour"].dtype) == "int16" and str(dl["day_of_week"].dtype) == "int16"
      and str(dl["month"].dtype) == "int16",
      f"{dl['hour'].dtype},{dl['day_of_week'].dtype},{dl['month'].dtype}")

# ---------------------------------------------------------------- split
check("train row count == 955625", len(tr) == 955_625, str(len(tr)))
check("test row count == 238975", len(te) == 238_975, str(len(te)))
check("train+test rows == demand_long rows",
      len(tr) + len(te) == len(dl),
      f"{len(tr)}+{len(te)} vs {len(dl)}")

key_tr = set(zip(tr["zone_id"], tr["time"]))
key_te = set(zip(te["zone_id"], te["time"]))
key_dl = set(zip(dl["zone_id"], dl["time"]))
check("train/test keys disjoint", key_tr.isdisjoint(key_te))
check("train union test == demand_long keys", (key_tr | key_te) == key_dl)

# per-zone chronological separation: max(train time) < min(test time)
tr_g = tr.groupby("zone_id")["time"]
te_g = te.groupby("zone_id")["time"]
tr_max = tr_g.max()
te_min = te_g.min()
zones_tr = set(tr_max.index)
zones_te = set(te_min.index)
check("every zone present in both train and test", zones_tr == zones_te)
common = sorted(zones_tr & zones_te)
viol = [zn for zn in common if not (tr_max[zn] < te_min[zn])]
check("train_max < test_min for EVERY zone", len(viol) == 0,
      f"violations={viol[:10]} n={len(viol)}")

# per-zone coverage: train+test counts equal per-zone total
per_zone_total = dl.groupby("zone_id").size()
tr_counts = tr.groupby("zone_id").size()
te_counts = te.groupby("zone_id").size()
reindex = sorted(per_zone_total.index)
cov = all(
    int(tr_counts.get(zn, 0)) + int(te_counts.get(zn, 0)) == int(per_zone_total[zn])
    for zn in reindex
)
check("per-zone train+test counts == per-zone total", cov)

# split metadata agrees with actual CSVs
meta_zones = set(meta["zone_id"])
check("split_metadata covers all 275 zones", len(meta_zones) == 275, str(len(meta_zones)))
meta_ok = True
for _, row in meta.iterrows():
    zn = row["zone_id"]
    if int(row["n_train"]) != int(tr_counts.get(zn, -1)) or int(row["n_test"]) != int(te_counts.get(zn, -1)):
        meta_ok = False
        break
check("split_metadata n_train/n_test match actual files", meta_ok)

# global summary cross-check against split_summary.json
import json
with (PROC / "split_summary.json").open(encoding="utf-8") as fh:
    summ = json.load(fh)
check("summary long_shape matches file", summ["long_shape"] == [len(dl), 3], str(summ["long_shape"]))
check("summary train_rows matches file", summ["train_rows"] == len(tr))
check("summary test_rows matches file", summ["test_rows"] == len(te))
check("summary n_zones matches file", summ["n_zones"] == dl["zone_id"].nunique())

# ---------------------------------------------------------------- report
print("=" * 78)
n_pass = sum(1 for _, ok, _ in results if ok)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print("=" * 78)
print(f"{n_pass}/{len(results)} checks passed")

sys.exit(0 if n_pass == len(results) else 1)
