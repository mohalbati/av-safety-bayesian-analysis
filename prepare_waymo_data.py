"""
Convert Waymo Safety Impact hub downloads into the compact inputs used by
compare_waymo.py and comparison_figs.py.

Usage:
    python prepare_waymo_data.py <CSV2 crash list> <CSV3 comparison table> <out_dir>

Inputs are the hub's files (data through June 2026):
    CSV2 - Crashes with SGO ID and Group Membership 202009-202606-2022benchmark.csv
    CSV3 - Collision Counts and Comparison to Benchmarks 202009-202606-2022benchmark.csv
Outputs (in <out_dir>):
    waymo_crash_list_compact.txt   (input to compare_waymo.py)
    waymo_csv3_key.csv             (input to comparison_figs.py)
"""
import re
import sys
from pathlib import Path
import pandas as pd

csv2, csv3, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

# ---- crash list -> compact pipe-delimited text
c2 = pd.read_csv(csv2, dtype=str, keep_default_na=False)
flags = [c for c in c2.columns if c.startswith("Is ")]
bits = c2[flags].apply(lambda r: "".join("1" if str(v).strip().lower() == "true" else "0" for v in r), axis=1)
cols = ["SGO Report ID", "SGO Report Version", "Year Month", "County", "Road Type", "Crash Type"]
lines = c2[cols].astype(str).agg("|".join, axis=1) + "|" + bits
(out / "waymo_crash_list_compact.txt").write_text(";".join(flags) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

# ---- comparison table -> key columns, all-crash rows only
c3 = pd.read_csv(csv3)
c3 = c3[c3["Grouping Type"] == "All Crashes"].copy()
ren = {"Outcome": "outcome", "Benchmark Comparison": "comparison", "County Name": "county", "Waymo Count": "waymo_count",
       "Waymo IPMM": "waymo_ipmm", "Waymo IPMM CI Lower": "waymo_lo", "Waymo IPMM CI Upper": "waymo_hi",
       "Benchmark IPMM": "bench_ipmm", "Benchmark IPMM CI Lower": "bench_lo", "Benchmark IPMM CI Upper": "bench_hi",
       "Predicted Benchmark Count": "pred_bench_count", "Waymo Percent Reduction": "red",
       "Waymo Percent Reduction CI Lower": "red_lo", "Waymo Percent Reduction CI Upper": "red_hi"}
k = c3[list(ren)].rename(columns=ren)
# "Any-injury-reported (Observed, Dynamic)" -> "Any-injury-reported (Observed Dynamic)"
k["comparison"] = k["comparison"].apply(lambda s: re.sub(r"\(([^)]*?), ([^)]*?)\)", r"(\1 \2)", s))
k.to_csv(out / "waymo_csv3_key.csv", index=False)
print(f"crash list: {len(c2)} rows -> {out / 'waymo_crash_list_compact.txt'}")
print(f"comparison table: {len(k)} all-crash rows -> {out / 'waymo_csv3_key.csv'}")
