"""
How much evidence is enough? A Bayesian rare-event view of driverless-vehicle
safety from NHTSA Standing General Order (SGO) crash reports.

Author: Mohammad Albati
Inputs : NHTSA SGO 2021-01 ADS incident reports (archive 2021-2025 + current 2025-),
         Waymo published rider-only (RO) miles and human benchmarks (through Jun 2026).
Usage  : python sgo_analysis.py <data_dir> <out_dir>
"""
import sys, re, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, GRAYM = "#0b0b0b", "#52514e", "#e6e5e1", "#9a998f"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRAYM,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "savefig.dpi": 200,
})

# ---------------------------------------------------------------- 1. load + harmonise
def load(fn, src):
    d = pd.read_csv(DATA / fn, encoding="latin1", low_memory=False)
    d["src"] = src
    return d

arch = load("ADS_archive.csv", "archive")
cur = load("SGO-2021-01_Incident_Reports_ADS.csv", "current")

# airbag: archive has separate SV / CP columns, current has one combined column
arch["airbag_any"] = (arch["SV Any Air Bags Deployed?"].eq("Yes") | arch["CP Any Air Bags Deployed?"].eq("Yes"))
arch["airbag_sv"] = arch["SV Any Air Bags Deployed?"].eq("Yes")
ab = cur["Any Air Bags Deployed?"].fillna("")
cur["airbag_any"] = ab.str.contains("Yes")
cur["airbag_sv"] = ab.str.contains("Yes Subject Vehicle")

d = pd.concat([arch, cur], ignore_index=True)
d["entity"] = d["Reporting Entity"].str.replace(r",? ?(LLC|Inc\.|Corp|Incorporated|Operations, Inc\.|LLC\.)$", "", regex=True).str.strip()
d["date"] = pd.to_datetime(d["Incident Date"].str.title(), format="%b-%Y", errors="coerce")
d["sub"] = pd.to_datetime(d["Report Submission Date"], format="mixed", errors="coerce")
d["ver"] = pd.to_numeric(d["Report Version"], errors="coerce")

# keep the latest version of each report (one report = one subject vehicle)
d = (d.sort_values(["Report ID", "ver", "sub"]).groupby("Report ID", as_index=False).tail(1))
d = d[d["Report Type"].ne("No New or Updated Incident Reports")]

sev = d["Highest Injury Severity Alleged"].fillna("Unknown")
d["inj_any"] = sev.str.contains("Minor|Moderate|Serious|Fatal", case=False)
d["inj_serious"] = sev.str.contains("Serious|Fatal", case=False)
d["inj_unknown"] = sev.str.contains("Unknown", case=False)

narr = d["Narrative"].fillna("").str.lower()
d["test_driver"] = narr.str.contains("test driver|safety driver|vehicle operator was|in the driver.?s seat")
d["remote_ctrl"] = narr.str.contains(r"remote (?:support|assistance|operator|driver|control|teleop)|remotely took control|teleoperat")
d["manual_mode"] = narr.str.contains("manual mode|disengaged|took over|took control")
d["human_action"] = d["remote_ctrl"] | d["manual_mode"] | narr.str.contains("safety monitor")
d["ads_not_engaged"] = d.get("Engagement Status", pd.Series(index=d.index, dtype=object)).fillna("").str.contains("Not Engaged") | narr.str.contains("ads was not engaged")
mv = d["SV Pre-Crash Movement"].fillna("Unknown")
d["sv_stationary"] = mv.isin(["Stopped", "Parked"])
d.to_csv(OUT / "sgo_ads_clean.csv", index=False)

# ---------------------------------------------------------------- 2. industry overview
top = d["entity"].value_counts().head(8).index.tolist()
ov = (d[d["entity"].isin(top)].groupby("entity")
      .agg(reports=("Report ID", "size"), any_injury=("inj_any", "sum"),
           serious_or_fatal=("inj_serious", "sum"), airbag_any=("airbag_any", "sum"),
           sv_stationary_share=("sv_stationary", "mean"),
           first=("date", "min"), last=("date", "max"))
      .sort_values("reports", ascending=False))
ov["sv_stationary_share"] = (ov["sv_stationary_share"] * 100).round(0)
ov["first"] = ov["first"].dt.strftime("%Y-%m"); ov["last"] = ov["last"].dt.strftime("%Y-%m")
ov.to_csv(OUT / "table1_operators.csv")

# ---------------------------------------------------------------- 3. Waymo rider-only subset
CUTOFF = pd.Timestamp("2026-06-01")  # Waymo exposure is through June 2026
TX_GA_CITIES = ["austin", "atlanta", "decatur", "sandy springs", "east point", "college park", "brookhaven",
                "round rock", "pflugerville", "del valle", "sunset valley", "west lake hills", "rollingwood"]
w = d[(d["entity"] == "Waymo") & (d["date"] <= CUTOFF)].copy()
st = w["State"].fillna("").str.strip()
city = w["City"].fillna("").str.strip().str.lower()
in_area = st.isin(["AZ", "CA"]) | (st.isin(["TX", "GA"]) & city.isin(TX_GA_CITIES))
ro = w[in_area & ~w["test_driver"] & ~w["ads_not_engaged"] & ~w["manual_mode"]].copy()

MILES = 271.3e6  # Waymo rider-only miles through June 2026 (waymo.com/safety/impact)
# Waymo-published rates, incidents per million miles (IPMM), all locations, through Jun 2026
PUB = {
    "Serious injury or worse": dict(col="inj_serious", waymo=0.01, bench=0.21, pub_red=94.82),
    "Airbag deployment (any vehicle)": dict(col="airbag_any", waymo=0.29, bench=1.61, pub_red=81.68),
    "Any injury reported": dict(col="inj_any", waymo=0.67, bench=3.77, pub_red=82.12),
}

# ---------------------------------------------------------------- 4. Bayesian Gamma-Poisson
# Jeffreys prior Gamma(0.5, 0) -> posterior Gamma(k + 0.5, rate = miles)
rows = []
rng = np.random.default_rng(7)
post_samples = {}
for name, p in PUB.items():
    k = int(ro[p["col"]].sum())
    m = MILES / 1e6
    post = stats.gamma(a=k + 0.5, scale=1 / m)
    lo, med, hi = post.ppf([0.025, 0.5, 0.975])
    b = p["bench"]
    s = post.rvs(200_000, random_state=rng)
    post_samples[name] = s
    red = (1 - s / b) * 100
    rows.append({
        "outcome": name, "SGO count (this analysis)": k,
        "Waymo-published IPMM": p["waymo"], "implied Waymo count": round(p["waymo"] * m),
        "posterior median IPMM": round(med, 3), "95% CrI low": round(lo, 3), "95% CrI high": round(hi, 3),
        "human benchmark IPMM": b, "P(rate < benchmark)": round(post.cdf(b), 6),
        "reduction median %": round(np.median(red), 1), "reduction 95% low %": round(np.percentile(red, 2.5), 1),
        "reduction 95% high %": round(np.percentile(red, 97.5), 1), "Waymo-published reduction %": p["pub_red"],
    })
bayes = pd.DataFrame(rows); bayes.to_csv(OUT / "table2_waymo_bayes.csv", index=False)

# sensitivity: benchmark uncertainty (lognormal, 20% and 40% CV) + 32% under-reporting already in any-injury bench
sens = []
for name, p in PUB.items():
    s = post_samples[name]
    for cv in [0.0, 0.2, 0.4]:
        sig = np.sqrt(np.log(1 + cv ** 2))
        bench = p["bench"] * np.exp(rng.normal(-sig ** 2 / 2, sig, s.size)) if cv else p["bench"]
        red = (1 - s / bench) * 100
        sens.append({"outcome": name, "benchmark CV": cv, "P(Waymo better)": round(float(np.mean(s < bench)), 4),
                     "reduction 2.5%": round(np.percentile(red, 2.5), 1), "reduction median": round(np.median(red), 1)})
pd.DataFrame(sens).to_csv(OUT / "table3_benchmark_sensitivity.csv", index=False)

# ---------------------------------------------------------------- 5. miles needed to demonstrate
def miles_needed(k, bench_ipmm, target_frac=1.0, conf=0.95):
    """Miles (millions) so that P(rate < target_frac * bench | k events) >= conf, Jeffreys prior."""
    q = stats.gamma(a=k + 0.5).ppf(conf)
    return q / (bench_ipmm * target_frac)

ks = np.arange(0, 11)
mn = pd.DataFrame({"events observed": ks})
for name, p in PUB.items():
    mn[f"{name}: better than human (M mi)"] = [round(miles_needed(k, p["bench"]), 1) for k in ks]
    mn[f"{name}: 50% better (M mi)"] = [round(miles_needed(k, p["bench"], 0.5), 1) for k in ks]
mn.to_csv(OUT / "table4_miles_to_demonstrate.csv", index=False)

# ---------------------------------------------------------------- 6. event tree: crash mechanism -> injury
et = []
for lab, mask in [("Waymo stationary (stopped/parked)", ro["sv_stationary"]),
                  ("Waymo moving", ~ro["sv_stationary"])]:
    sub = ro[mask]
    n, k = len(sub), int(sub["inj_any"].sum())
    bp = stats.beta(k + 0.5, n - k + 0.5)
    et.append({"branch": lab, "crashes": n, "share of crashes %": round(100 * n / len(ro), 1),
               "injury crashes": k, "P(injury | branch) median %": round(100 * bp.median(), 1),
               "95% low %": round(100 * bp.ppf(0.025), 1), "95% high %": round(100 * bp.ppf(0.975), 1),
               "share of injury crashes %": round(100 * k / max(ro["inj_any"].sum(), 1), 1)})
et = pd.DataFrame(et); et.to_csv(OUT / "table5_event_tree.csv", index=False)

partner = (ro.assign(p=ro["Crash With"].fillna("Unknown").replace({
    "Passenger Car": "Car / SUV / pickup / van", "SUV": "Car / SUV / pickup / van",
    "Pickup Truck": "Car / SUV / pickup / van", "Van": "Car / SUV / pickup / van",
    "Heavy Truck": "Heavy truck / bus", "Bus": "Heavy truck / bus",
    "Non-Motorist: Cyclist": "Cyclist / pedestrian / scooter", "Non-Motorist: Pedestrian": "Cyclist / pedestrian / scooter",
    "Non-Motorist: Scooter - Skateboard": "Cyclist / pedestrian / scooter", "Non-Motorist: Other": "Cyclist / pedestrian / scooter",
    "Non-Motorist: Wheelchair": "Cyclist / pedestrian / scooter"}))
    .groupby("p").agg(crashes=("Report ID", "size"), injury=("inj_any", "sum")).sort_values("crashes", ascending=False))
partner.to_csv(OUT / "table6_waymo_crash_partner.csv")

# ---------------------------------------------------------------- 7. human-in-the-loop across operators
hitl = (d[d["entity"].isin(top) & (d["date"] >= "2025-01-01")].groupby("entity")
        .agg(reports=("Report ID", "size"), remote_operator_in_narrative=("remote_ctrl", "sum"),
             test_driver_present=("test_driver", "sum"), manual_or_takeover=("manual_mode", "sum"), any_human_action=("human_action", "sum")))
hitl["human action share %"] = (100 * hitl["any_human_action"] / hitl["reports"]).round(1)
hitl["remote share %"] = (100 * hitl["remote_operator_in_narrative"] / hitl["reports"]).round(1)
hitl.sort_values("reports", ascending=False).to_csv(OUT / "table7_human_in_loop_2025on.csv")

# ---------------------------------------------------------------- charts
# Fig 1: Waymo posterior vs benchmark (dot + whisker, log x)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
y = np.arange(len(bayes))[::-1]
for yi, (_, r) in zip(y, bayes.iterrows()):
    ax.plot([r["95% CrI low"], r["95% CrI high"]], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(r["posterior median IPMM"], yi, "o", color=BLUE, ms=8, mec="#fcfcfb", mew=2, zorder=3)
    ax.plot(r["human benchmark IPMM"], yi, "D", color=INK2, ms=7, mec="#fcfcfb", mew=1.5, zorder=3)
    ax.annotate(f'{r["reduction median %"]:.0f}% lower\n(95% CrI {r["reduction 95% low %"]:.0f}-{r["reduction 95% high %"]:.0f}%)',
                (r["human benchmark IPMM"], yi), xytext=(10, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK2)
ax.set_yticks(y, bayes["outcome"]); ax.set_xscale("log"); ax.set_xlim(0.004, 30)
ax.set_xlabel("Crashed vehicles per million miles (log scale)")
ax.set_ylim(-0.6, 2.6)
ax.set_title("Waymo rider-only crash rates are 84-89% below the human benchmark")
ax.plot([], [], "o", color=BLUE, label="Waymo posterior median + 95% credible interval")
ax.plot([], [], "D", color=INK2, label="Human benchmark (same cities)")
ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.52), ncol=2, frameon=False, fontsize=8.5)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig1_waymo_vs_benchmark.png", bbox_inches="tight"); plt.close(fig)

# Fig 2: miles needed vs events observed (serious-injury outcome)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
b = PUB["Serious injury or worse"]["bench"]
ax.plot(ks, [miles_needed(k, b) for k in ks], "-o", color=BLUE, lw=2, ms=6, label="Claim: better than human")
ax.plot(ks, [miles_needed(k, b, 0.5) for k in ks], "-o", color=ORANGE, lw=2, ms=6, label="Claim: at least 50% better")
ax.axhline(MILES / 1e6, color=GRAYM, lw=1, ls="--")
ax.text(10, MILES / 1e6 * 0.96, "Waymo RO miles to date (271M)", ha="right", va="top", fontsize=8.5, color=INK2)
for k_, lab in [(5, "5 events")]:
    v = miles_needed(k_, b)
    ax.annotate(f"{v:.0f}M mi", (k_, v), xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK2)
ax.set_xlabel("Serious-or-fatal injury crashes observed"); ax.set_ylabel("Miles needed (millions)")
ax.set_xticks(ks)
ax.set_title("A new operator needs ~9M crash-free miles to claim it beats humans on serious injuries")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig2_miles_to_demonstrate.png", bbox_inches="tight"); plt.close(fig)

# Fig 3: quarterly SGO reports, four distinct operators (GM duplicates Cruise reports, so it is left out)
ops = ["Waymo", "Zoox", "Cruise", "Tesla"]
mon = (d[d["entity"].isin(ops) & d["date"].notna()].groupby([pd.Grouper(key="date", freq="QS"), "entity"])
       .size().unstack(fill_value=0))
mon = mon[mon.index < "2026-07-01"]
fig, ax = plt.subplots(figsize=(7.2, 3.1))
for col, c in zip(ops, [BLUE, ORANGE, AQUA, YELLOW]):
    if col in mon:
        ax.plot(mon.index, mon[col], color=c, lw=2, label=col)
ax.axvline(pd.Timestamp("2025-06-15"), color=GRAYM, lw=1, ls="--")
ax.text(pd.Timestamp("2025-06-01"), 330, "Third amended SGO\n(reporting rules change)", ha="right", va="top", fontsize=8, color=INK2)
ax.set_ylabel("Reports per quarter"); ax.set_ylim(0, 345)
ax.set_title("Waymo files most SGO crash reports, in step with its fast-growing mileage")
ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig3_reports_by_quarter.png", bbox_inches="tight"); plt.close(fig)

# Fig 4: event tree branch - P(injury | crash partner), Beta posterior
pt = partner.drop(index=[i for i in partner.index if i in ("Other, see Narrative", "Pole / Tree", "Animal", "First Responder Vehicle")])
pt = pt.assign(med=lambda t: [100 * stats.beta(k + .5, n - k + .5).median() for n, k in zip(t.crashes, t.injury)],
               lo=lambda t: [100 * stats.beta(k + .5, n - k + .5).ppf(.025) for n, k in zip(t.crashes, t.injury)],
               hi=lambda t: [100 * stats.beta(k + .5, n - k + .5).ppf(.975) for n, k in zip(t.crashes, t.injury)])
pt = pt.sort_values("med")
pt.to_csv(OUT / "table8_injury_given_partner.csv")
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for yi, (lab, r) in enumerate(pt.iterrows()):
    ax.plot([r.lo, r.hi], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(r.med, yi, "o", color=BLUE, ms=8, mec="#fcfcfb", mew=2, zorder=3)
    ax.annotate(f"{int(r.injury)} of {int(r.crashes)}", (r.hi, yi), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(pt)), pt.index); ax.set_xlim(0, 80)
ax.set_xlabel("P(injury | crash), %  -  median and 95% credible interval")
ax.set_title("Half of Waymo crashes with a cyclist or pedestrian injure someone")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig4_injury_given_partner.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 8. city-level posteriors (Waymo per-city miles and benchmarks)
CITY = {  # RO miles (M) through Jun 2026 and Waymo headline human benchmarks (IPMM), Safety Impact CSV3
    "Phoenix":       dict(miles=92.121, inj_any=1.9539, airbag_any=1.3227, inj_serious=0.10353),
    "San Francisco": dict(miles=82.421, inj_any=6.6394, airbag_any=1.9189, inj_serious=0.39072),
    "Los Angeles":   dict(miles=67.098, inj_any=2.5537, airbag_any=1.2659, inj_serious=0.15617),
    "Austin":        dict(miles=21.064, inj_any=3.2554, airbag_any=2.3233, inj_serious=0.17493),
    "Atlanta":       dict(miles=8.624,  inj_any=6.5803, airbag_any=2.8312, inj_serious=0.22864),
}
# latitude/zip are redacted in SGO, so CA reports are split into metros by city name
LA_CITIES = {"los angeles", "santa monica", "culver city", "inglewood", "beverly hills", "west hollywood", "venice",
             "marina del rey", "lennox", "carson", "harbor city", "el segundo", "hawthorne"}
SF_CITIES = {"san francisco", "mountain view", "daly city", "south san francisco", "palo alto", "redwood city",
             "san mateo", "sunnyvale", "san bruno", "brisbane", "san jose", "santa clara", "burlingame",
             "portola valley", "menlo park", "emeryville", "millbrae", "los altos", "foster city"}
st_ro = ro["State"].fillna("").str.strip(); cty = ro["City"].fillna("").str.strip().str.lower()
ro["metro"] = np.select([st_ro.eq("AZ"), st_ro.eq("CA") & cty.isin(SF_CITIES), st_ro.eq("CA") & cty.isin(LA_CITIES),
                         st_ro.eq("TX"), st_ro.eq("GA")],
                        ["Phoenix", "San Francisco", "Los Angeles", "Austin", "Atlanta"], "Other")
ro["metro"].value_counts().to_csv(OUT / "table9b_metro_counts.csv")
crow = []
for c, v in CITY.items():
    sub = ro[ro["metro"] == c]
    for col, lab in [("inj_any", "Any injury"), ("airbag_any", "Airbag (any vehicle)"), ("inj_serious", "Serious or worse")]:
        k = int(sub[col].sum()); post = stats.gamma(a=k + .5, scale=1 / v["miles"])
        sm = post.rvs(100_000, random_state=rng); red = (1 - sm / v[col]) * 100
        crow.append({"metro": c, "outcome": lab, "RO miles (M)": v["miles"], "crashes (all)": len(sub), "events k": k,
                     "posterior median IPMM": round(post.median(), 3), "95% low": round(post.ppf(.025), 3),
                     "95% high": round(post.ppf(.975), 3), "benchmark IPMM": v[col],
                     "P(better)": round(post.cdf(v[col]), 4), "reduction median %": round(np.median(red), 1),
                     "reduction 95% low %": round(np.percentile(red, 2.5), 1)})
city_t = pd.DataFrame(crow); city_t.to_csv(OUT / "table9_city_posteriors.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 3.1))
ci = city_t[city_t["outcome"] == "Any injury"].reset_index(drop=True)
yy = np.arange(len(ci))[::-1]
for yi, (_, r) in zip(yy, ci.iterrows()):
    ax.plot([r["95% low"], r["95% high"]], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(r["posterior median IPMM"], yi, "o", color=BLUE, ms=8, mec="#fcfcfb", mew=2, zorder=3)
    ax.plot(r["benchmark IPMM"], yi, "D", color=INK2, ms=7, mec="#fcfcfb", mew=1.5, zorder=3)
    ax.annotate(f'{r["reduction median %"]:.0f}% lower  ({int(r["events k"])} events, {r["RO miles (M)"]:.0f}M mi)',
                (r["benchmark IPMM"], yi), xytext=(10, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(yy, ci["metro"]); ax.set_xlim(0, 13); ax.set_ylim(-0.6, len(ci) - 0.4)
ax.set_xlabel("Any-injury crashed vehicles per million miles")
ax.set_title("The any-injury reduction holds in every city, from 67% in LA to 91% in SF")
ax.plot([], [], "o", color=BLUE, label="Waymo posterior median + 95% CrI"); ax.plot([], [], "D", color=INK2, label="Human benchmark (city)")
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig5_city_any_injury.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 9. severity trend: P(injury | crash) by year
yr = []
for y_, sub in ro.groupby(ro["date"].dt.year):
    n, k = len(sub), int(sub["inj_any"].sum()); bp = stats.beta(k + .5, n - k + .5)
    yr.append({"year": int(y_), "crashes": n, "injury crashes": k, "P(injury|crash) median %": round(100 * bp.median(), 1),
               "95% low %": round(100 * bp.ppf(.025), 1), "95% high %": round(100 * bp.ppf(.975), 1),
               "share stationary %": round(100 * sub["sv_stationary"].mean(), 1)})
pd.DataFrame(yr).to_csv(OUT / "table10_severity_by_year.csv", index=False)

summary = {"reports_after_dedup": int(len(d)), "date_min": str(d["date"].min().date()), "date_max": str(d["date"].max().date()),
           "operators": int(d["entity"].nunique()), "waymo_ro_crashes_thru_jun2026": int(len(ro)),
           "waymo_reports_total": int((d["entity"] == "Waymo").sum())}
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
for f in sorted(OUT.glob("table*.csv")):
    print("\n#", f.name); print(pd.read_csv(f).to_string(index=False))
