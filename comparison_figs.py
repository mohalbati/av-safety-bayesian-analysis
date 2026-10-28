"""
Comparison with Waymo's published methodology + concept / uncertainty figures.
Usage: python comparison_figs.py <out_dir> <waymo_csv3_key.csv>
Requires sgo_analysis.py and compare_waymo.py outputs in <out_dir>.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1]); W3 = pd.read_csv(sys.argv[2])
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
INK, INK2, GRID, GRAYM, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#9a998f", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRAYM, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200})
rng = np.random.default_rng(11)
M = 271.329  # Waymo RO miles (M), CSV1 "All Locations (mileage blended)"

OUTC = [  # label, our k, Waymo k, Waymo headline benchmark, Waymo published reduction + CI
    ("Serious injury or worse", 6, 3, 0.21331, -0.94817, -0.99181, -0.83157),
    ("Airbag deployment (any vehicle)", 72, 80, 1.6154, -0.81748, -0.86018, -0.76605),
    ("Any injury reported", 166, 183, 3.7736, -0.82127, -0.8496, -0.78925),
]
COL = {OUTC[0][0]: MAGENTA, OUTC[1][0]: ORANGE, OUTC[2][0]: BLUE}
N = 400_000
rows = []

def post(k, m): return stats.gamma(a=k + .5, scale=1 / m)

# ------------------------------------------------------------------ Fig A: workflow concept
fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
def box(x, y, w, h, t, fc="#eef3fb", ec=BLUE, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal", wrap=True)
def arrow(x1, y1, x2, y2, c=GRAYM):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))
box(1, 35, 19, 11, "NHTSA SGO ADS\nreports\n3,890 rows", bold=True)
box(25, 35, 19, 11, "De-duplicate\n(latest version)\n3,450 reports")
box(49, 35, 21, 11, "Waymo rider-only\nfilter (area, date,\nno test driver)\n1,934 vehicles")
box(75, 35, 23, 11, "Outcome counts k\nany injury 166\nairbag 72 · serious 6")
box(1, 18, 19, 11, "Waymo exposure\n271.3M RO miles\n+ benchmarks", fc="#fdf1ea", ec=ORANGE, fs=7.6)
box(25, 18, 45, 11, "Bayesian rare-event model\nGamma–Poisson rates · Beta–binomial branches\nMonte Carlo benchmark-uncertainty propagation", fc="#e9f6f0", ec=AQUA, fs=7.6)
box(75, 18, 23, 11, "Waymo published\ncrash list + CIs\n(Safety Impact hub)", fc="#fdf1ea", ec=ORANGE, fs=7.6)
box(1, 1, 97, 11, "Outputs: posterior rates and reductions · P(better than human) · exceedance percentiles\n"
                  "benchmark sensitivity · miles-to-demonstrate and power · event tree · human-in-the-loop · reconciliation",
    fc="#f3f2ee", ec=GRAYM, fs=7.6)
arrow(20.5, 40.5, 24.5, 40.5); arrow(44.5, 40.5, 48.5, 40.5); arrow(70.5, 40.5, 74.5, 40.5)
arrow(86.5, 34.5, 60, 29.5); arrow(20.5, 23.5, 24.5, 23.5); arrow(86.5, 29.5, 86.5, 34.5, c=ORANGE)
arrow(47.5, 17.5, 47.5, 12.5)
ax.text(87.5, 31.6, "reconcile", fontsize=7.5, color=ORANGE, ha="left")
ax.set_title("Analysis workflow: from raw crash reports to decision-ready risk metrics")
fig.tight_layout(); fig.savefig(OUT / "figA_workflow.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig B: Bayesian updating concept
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, (k, m, b, ttl) in zip(axs, [(6, M, 0.21331, "All locations: k = 6 in 271.3M mi"),
                                     (0, 8.624, 0.22864, "Atlanta: k = 0 in 8.6M mi")]):
    x = np.linspace(1e-4, 0.6 if k == 0 else 0.25, 1200)
    pr = x ** -0.5; pr /= np.trapezoid(pr, x)
    lk = stats.poisson.pmf(k, x * m); lk /= np.trapezoid(lk, x)
    po = post(k, m).pdf(x)
    top = 1.35 * max(lk[x > 0.004].max(), po[x > 0.004].max())
    ax.plot(x, pr, color=GRAYM, lw=1.6, ls="--", label="Prior (Jeffreys, rescaled)")
    ax.plot(x, lk, color=ORANGE, lw=1.6, label="Likelihood (normalized)")
    ax.plot(x, po, color=BLUE, lw=2.2, label="Posterior")
    ax.fill_between(x, 0, po, where=x < b, color=BLUE, alpha=.12)
    ax.set_ylim(0, top); ax.axvline(b, color=INK2, lw=1.2); ax.text(b, top * .97, " human\n benchmark", fontsize=7.5, color=INK2, va="top")
    ax.set_title(ttl, fontsize=9.5); ax.set_xlabel("Serious-injury rate, per million miles"); ax.set_yticks([])
    ax.text(.97, .72, f"P(better) = {post(k, m).cdf(b):.3f}", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK)
axs[0].legend(frameon=False, fontsize=7.2, loc="center right", bbox_to_anchor=(0.97, 0.38))
fig.suptitle("Bayesian updating: the posterior combines a weak prior with the observed count and exposure",
             x=0.01, ha="left", fontsize=10.5, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT / "figB_bayes_concept.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig C: posterior densities with percentile bands
fig, axs = plt.subplots(3, 1, figsize=(7.4, 5.6))
for ax, (lab, k, kw, b, *_r) in zip(axs, OUTC):
    p = post(k, M); q = p.ppf([.025, .25, .5, .75, .975])
    x = np.linspace(p.ppf(1e-5), max(b * 1.15, p.ppf(.99999)), 1500); y = p.pdf(x)
    ax.plot(x, y, color=COL[lab], lw=2)
    ax.fill_between(x, 0, y, where=(x >= q[0]) & (x <= q[4]), color=COL[lab], alpha=.18)
    ax.fill_between(x, 0, y, where=(x >= q[1]) & (x <= q[3]), color=COL[lab], alpha=.32)
    ax.axvline(q[2], color=COL[lab], lw=1, ls=":")
    ax.axvline(b, color=INK2, lw=1.4)
    ax.text(b, y.max() * .9, "  human benchmark\n  " + f"{b:.2f} IPMM", fontsize=7.8, color=INK2, va="top")
    ax.text(q[2], y.max() * 1.02, f"median {q[2]:.3f}", fontsize=7.8, color=INK2, ha="center")
    ax.set_title(f"{lab}: k = {k};  50% interval {q[1]:.3f}–{q[3]:.3f},  95% interval {q[0]:.3f}–{q[4]:.3f} IPMM", fontsize=9)
    ax.set_yticks([]); ax.set_ylim(0, y.max() * 1.15)
    rows.append({"outcome": lab, "k": k, "p2.5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p97.5": q[4],
                 "P(rate<bench)": p.cdf(b)})
axs[-1].set_xlabel("Crashed vehicles per million miles")
fig.suptitle("Posterior uncertainty distributions: dark band = 50% interval, light band = 95% interval",
             x=0.01, ha="left", fontsize=10.5, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT / "figC_posterior_densities.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(rows).to_csv(OUT / "table14_posterior_percentiles.csv", index=False)

# ------------------------------------------------------------------ Fig D: reduction distributions vs Waymo published CI
rd = []
fig, axs = plt.subplots(3, 1, figsize=(7.4, 5.8), sharex=False)
for ax, (lab, k, kw, b, wr, wlo, whi) in zip(axs, OUTC):
    for kk, c, name in [(k, COL[lab], f"This analysis (k = {k})"), (kw, INK2, f"Bayesian with Waymo's count (k = {kw})")]:
        s = 100 * (1 - post(kk, M).rvs(N, random_state=rng) / b)
        lo_, md, hi_ = np.percentile(s, [2.5, 50, 97.5])
        kde = stats.gaussian_kde(s[:40000]); xx = np.linspace(max(s.min(), 40), 100, 600)
        yy = kde(xx); ax.plot(xx, yy, color=c, lw=2 if kk == k else 1.4, ls="-" if kk == k else "--", label=name)
        rd.append({"outcome": lab, "counts": name, "median %": md, "2.5%": lo_, "97.5%": hi_,
                   "P(reduction>50%)": float(np.mean(s > 50)), "P(reduction>75%)": float(np.mean(s > 75))})
    ymax = ax.get_ylim()[1]
    ax.errorbar(-100 * wr, ymax * .55, xerr=[[-100 * wr + 100 * whi], [-100 * wlo + 100 * wr]], fmt="D", color=ORANGE,
                ms=6, capsize=3, lw=1.6, label="Waymo published: point + 95% CI (Clopper–Pearson)")
    ax.set_title(lab, fontsize=9.5); ax.set_yticks([]); ax.set_xlim(55 if k > 10 else 40, 100)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
axs[-1].set_xlabel("Reduction in crash rate relative to the human benchmark (%)")
fig.suptitle("Reduction distributions: our posterior, the posterior with Waymo's counts, and Waymo's published interval",
             x=0.01, ha="left", fontsize=10.2, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT / "figD_reduction_vs_waymo.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(rd).to_csv(OUT / "table15_reduction_percentiles.csv", index=False)

# ------------------------------------------------------------------ Fig E: exceedance curves P(R > x)
fig, ax = plt.subplots(figsize=(7.4, 3.3))
xs = np.linspace(0, 100, 501); ex = []
for lab, k, kw, b, *_r in OUTC:
    s = 100 * (1 - post(k, M).rvs(N, random_state=rng) / b)
    pe = [(s > x).mean() for x in xs]; ax.plot(xs, pe, color=COL[lab], lw=2, label=lab)
    for pr_ in [0.95, 0.5, 0.05]:
        ex.append({"outcome": lab, "exceedance probability": pr_, "reduction at least %": float(np.percentile(s, 100 * (1 - pr_)))})
for yv in [0.95, 0.5, 0.05]: ax.axhline(yv, color=GRAYM, lw=.8, ls=":")
ax.text(1, .955, "95% exceedance (P5)", fontsize=7.5, color=INK2, va="bottom")
ax.text(1, .505, "median (P50)", fontsize=7.5, color=INK2, va="bottom")
ax.text(1, .055, "5% exceedance (P95)", fontsize=7.5, color=INK2, va="bottom")
ax.set_xlabel("Reduction threshold x (%)"); ax.set_ylabel("P(reduction > x)"); ax.set_xlim(0, 100); ax.set_ylim(0, 1.02)
ax.set_title("Exceedance curves: the probability that the true reduction exceeds any chosen threshold")
ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.12))
fig.tight_layout(); fig.savefig(OUT / "figE_exceedance.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(ex).to_csv(OUT / "table16_exceedance.csv", index=False)

# ------------------------------------------------------------------ Fig F: overlap with uncertain benchmark
ovl = []
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
for ax, (lab, k, kw, b, *_r) in zip(axs, OUTC):
    p = post(k, M); hi = b * 3.2
    x = np.linspace(1e-5, hi, 3000); yw = p.pdf(x)
    for cv, ls_ in [(0.2, "--"), (0.4, "-")]:
        sg = np.sqrt(np.log(1 + cv ** 2)); bh = stats.lognorm(s=sg, scale=b * np.exp(-sg ** 2 / 2))
        yh = bh.pdf(x); ov = np.trapezoid(np.minimum(yw, yh), x)
        ax.plot(x, yh / yh.max(), color=INK2, lw=1.3, ls=ls_, label=f"benchmark, CV {cv:.0%}")
        ovl.append({"outcome": lab, "benchmark CV": cv, "overlap coefficient": ov,
                    "P(Waymo rate > benchmark draw)": float(np.mean(p.rvs(N, random_state=rng) > bh.rvs(N, random_state=rng)))})
    ax.plot(x, yw / yw.max(), color=COL[lab], lw=2, label="Waymo posterior")
    ax.set_ylim(0, 1.12); ax.set_yticks([])
    ax.set_title(lab.replace(" (any vehicle)", "").replace(" reported", ""), fontsize=9)
    ax.set_xlabel("IPMM")
axs[0].legend(frameon=False, fontsize=6.8, loc="upper right")
fig.suptitle("Overlap between Waymo's posterior and an uncertain human benchmark (peaks scaled to 1)",
             x=0.01, ha="left", fontsize=10.2, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT / "figF_overlap.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(ovl).to_csv(OUT / "table17_overlap.csv", index=False)

# ------------------------------------------------------------------ Fig G: benchmark model-form uncertainty (any injury)
var = W3[(W3.outcome == "Any Injury") & (W3.county == "All Locations (mileage blended)")].copy()
LAB = {"Any-injury-reported (observed non-Dynamic)": "Observed, unadjusted",
       "Any-injury-reported (Observed Dynamic)": "Observed, spatially adjusted",
       "Any-injury-reported (non-Dynamic)": "Under-reporting corrected, unadjusted",
       "Any-injury-reported": "Under-reporting corrected, spatially adjusted (headline)"}
var["label"] = var.comparison.map(LAB); var = var.sort_values("bench_ipmm")
fig, ax = plt.subplots(figsize=(7.4, 3.4)); mix = []; mf = []
cols = [YELLOW, AQUA, VIOLET, BLUE]
for c, (_, r) in zip(cols, var.iterrows()):
    s = 100 * (1 - post(183, M).rvs(N // 4, random_state=rng) / r.bench_ipmm); mix.append(s)
    kde = stats.gaussian_kde(s[:30000]); xx = np.linspace(55, 92, 500)
    ax.plot(xx, kde(xx), color=c, lw=1.8, label=f"{r.label}: benchmark {r.bench_ipmm:.2f}")
    mf.append({"benchmark variant": r.label, "benchmark IPMM": r.bench_ipmm, "Waymo published reduction %": -100 * r.red,
               "Waymo CI low %": -100 * r.red_hi, "Waymo CI high %": -100 * r.red_lo,
               "Bayes median %": np.median(s), "Bayes 2.5%": np.percentile(s, 2.5), "Bayes 97.5%": np.percentile(s, 97.5)})
allm = np.concatenate(mix); kde = stats.gaussian_kde(allm[::8])
ax.plot(xx, kde(xx), color=INK, lw=2.4, ls="--", label="Equal-weight mixture (bounding range)")
mf.append({"benchmark variant": "Equal-weight mixture", "benchmark IPMM": np.nan, "Waymo published reduction %": np.nan,
           "Waymo CI low %": np.nan, "Waymo CI high %": np.nan, "Bayes median %": np.median(allm),
           "Bayes 2.5%": np.percentile(allm, 2.5), "Bayes 97.5%": np.percentile(allm, 97.5)})
ax.set_yticks([]); ax.set_xlabel("Any-injury reduction (%), Waymo count k = 183")
ax.set_title("Benchmark choice moves the any-injury reduction more than sampling noise does")
ax.legend(frameon=False, fontsize=7.3, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "figG_benchmark_modelform.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(mf).to_csv(OUT / "table18_benchmark_variants.csv", index=False)

# ------------------------------------------------------------------ Fig H: frequentist (Waymo) vs Bayesian intervals by city
MI = {"Maricopa": ("Phoenix", 92.121), "San Francisco Bay Area": ("SF Bay Area", 82.421), "Los Angeles": ("Los Angeles", 67.098),
      "Travis": ("Austin", 21.064), "Atlanta Area": ("Atlanta", 8.624), "All Locations (mileage blended)": ("All locations", M)}
HEAD = {"Any Injury": "Any-injury-reported", "Any Airbag Deployment": "Any Airbag Deployment",
        "Suspected Serious Injury+": "Suspected Serious Injury+"}
fb = []
for o, comp in HEAD.items():
    for cty, (nm, mi) in MI.items():
        r = W3[(W3.outcome == o) & (W3.comparison == comp) & (W3.county == cty)].iloc[0]
        s = 100 * (1 - post(int(r.waymo_count), mi).rvs(N // 2, random_state=rng) / r.bench_ipmm)
        fb.append({"outcome": o, "location": nm, "k": int(r.waymo_count), "miles M": mi,
                   "Waymo point %": -100 * r.red, "Waymo low %": -100 * r.red_hi, "Waymo high %": -100 * r.red_lo,
                   "Bayes median %": np.median(s), "Bayes low %": np.percentile(s, 2.5), "Bayes high %": np.percentile(s, 97.5),
                   "P(better)": float(np.mean(s > 0))})
fb = pd.DataFrame(fb); fb.to_csv(OUT / "table19_freq_vs_bayes.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.6), sharey=True)
for ax, o in zip(axs, HEAD):
    sub = fb[fb.outcome == o].reset_index(drop=True); yy = np.arange(len(sub))[::-1]
    for yi, (_, r) in zip(yy, sub.iterrows()):
        ax.plot([r["Waymo low %"], r["Waymo high %"]], [yi + .14] * 2, color=ORANGE, lw=2)
        ax.plot(r["Waymo point %"], yi + .14, "D", color=ORANGE, ms=4.5)
        ax.plot([r["Bayes low %"], r["Bayes high %"]], [yi - .14] * 2, color=BLUE, lw=2)
        ax.plot(r["Bayes median %"], yi - .14, "o", color=BLUE, ms=4.5)
        ax.text(108, yi, f"k={r.k}", fontsize=7, color=INK2, va="center")
    ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-120, 135); ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks(yy, sub.location); ax.set_title(o.replace("Suspected Serious Injury+", "Serious injury+"), fontsize=9)
    ax.set_xlabel("Reduction (%)"); ax.grid(axis="y", visible=False)
axs[0].plot([], [], "D-", color=ORANGE, label="Waymo: Clopper–Pearson 95% CI")
axs[0].plot([], [], "o-", color=BLUE, label="Bayesian 95% CrI (same counts)")
fig.legend(frameon=False, fontsize=7.8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Same data, two inference methods: intervals agree when counts are large and diverge at k = 0",
             x=0.01, ha="left", fontsize=10.2, weight="bold", color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "figH_freq_vs_bayes.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig I: power / evidence accumulation
b = 0.21331
def kcrit(m):
    k = -1
    while post(k + 1, m).cdf(b) >= 0.95: k += 1
    return k
ms = np.linspace(0.5, 120, 240); pw = []
fig, ax = plt.subplots(figsize=(7.4, 3.2))
for r_, c in [(0.0, GRAYM), (0.5, YELLOW), (0.8, ORANGE), (0.9, BLUE)]:
    lam = (1 - r_) * b
    pp = [stats.poisson.cdf(kcrit(m), lam * m) if kcrit(m) >= 0 else 0.0 for m in ms]
    ax.plot(ms, pp, color=c, lw=2, label=f"true reduction {r_:.0%}")
    for target in [0.5, 0.8]:
        hit = next((m for m, v in zip(ms, pp) if v >= target), np.nan); pw.append({"true reduction": r_, "power": target, "miles M": hit})
ax.axhline(0.8, color=GRAYM, lw=.8, ls=":"); ax.text(119, 0.81, "80% power", ha="right", fontsize=7.5, color=INK2)
ax.set_xlabel("Rider-only miles (millions)"); ax.set_ylabel("P(can claim better than human)"); ax.set_ylim(0, 1)
ax.set_title("Power to demonstrate a serious-injury benefit grows slowly and in steps with exposure")
ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "figI_power.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(pw).to_csv(OUT / "table20_power_miles.csv", index=False)

# ------------------------------------------------------------------ Fig J: reconciliation with Waymo's crash list
ov = pd.read_csv(OUT / "table11_set_overlap.csv"); ag = pd.read_csv(OUT / "table12_outcome_agreement.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.35]})
v = dict(zip(ov.set.str.strip(), ov.vehicles))
cats = ["In both", "Only in this analysis", "Only in Waymo list"]; vals = [v[c] for c in cats]
axs[0].barh(cats[::-1], vals[::-1], color=[GRAYM, ORANGE, BLUE][::-1], height=.6)
for i, val in enumerate(vals[::-1]): axs[0].text(val + 20, i, f"{val:,}", va="center", fontsize=8.5, color=INK)
axs[0].set_xlim(0, 2300); axs[0].set_title("Crashed-vehicle sets (5 metros)", fontsize=9); axs[0].grid(axis="y", visible=False)
labs = ag.outcome.tolist(); x = np.arange(len(labs))
axs[1].bar(x - .2, ag["only ours"], width=.38, color=ORANGE, label="flagged only by this analysis")
axs[1].bar(x + .2, ag["only Waymo"], width=.38, color=BLUE, label="flagged only by Waymo")
notes = ["17 of 20: 'Unknown' severity\nwith injury narrative", "9 airbags, consistent\nwith video review", "3 SGO 'Serious' outside\nWaymo's police-confirmed set"]
for i, n in enumerate(notes): axs[1].text(i, max(ag["only ours"][i], ag["only Waymo"][i]) + 1, n, ha="center", fontsize=7, color=INK2)
axs[1].set_xticks(x, ["Any injury", "Airbag", "Serious+"]); axs[1].set_ylim(0, 36)
axs[1].set_title("Outcome flags on shared vehicles", fontsize=9); axs[1].legend(frameon=False, fontsize=7.2, loc="upper right", bbox_to_anchor=(1, 1.02))
fig.suptitle("Reconciliation: 99.5% of vehicles match Waymo's list; outcome gaps are consistent with Waymo's manual review",
             x=0.01, ha="left", fontsize=10, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT / "figJ_reconciliation.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig K: event tree with numbers
et = pd.read_csv(OUT / "table5_event_tree.csv")
lam_c = post(1934, M); fig, ax = plt.subplots(figsize=(7.4, 3.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
def node(x, y, t, fc="#eef3fb", ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y - 4), 22, 8, boxstyle="round,pad=0.3,rounding_size=1", fc=fc, ec=ec, lw=1.1))
    ax.text(x + 11, y, t, ha="center", va="center", fontsize=7.6, color=INK)
node(1, 30, f"Initiating event:\ncrash, {1934/M:.2f} per M mi")
ys = {"Waymo stationary (stopped/parked)": 45, "Waymo moving": 15}; end = []
for _, r in et.iterrows():
    y0 = ys[r.branch]; share = r["share of crashes %"] / 100
    ax.plot([23.5, 34], [30, y0], color=GRAYM, lw=1.2)
    node(34, y0, f"{'Stationary' if 'stationary' in r.branch else 'Moving'}\nP = {share:.3f}")
    pinj = r["P(injury | branch) median %"] / 100
    for dy, lab_, pp, c in [(6, "Injury", pinj, ORANGE), (-6, "No injury", 1 - pinj, AQUA)]:
        ax.plot([56.5, 67], [y0, y0 + dy], color=GRAYM, lw=1.2)
        rate = 1934 / M * share * pp
        ax.add_patch(FancyBboxPatch((67, y0 + dy - 3), 31, 6, boxstyle="round,pad=0.3,rounding_size=1", fc="#ffffff", ec=c, lw=1.1))
        ax.text(82.5, y0 + dy, f"{lab_}: P = {pp:.3f}  →  {rate:.2f} per M mi", ha="center", va="center", fontsize=7.4, color=INK)
        end.append({"branch": r.branch, "end state": lab_, "conditional P": pp, "end-state rate IPMM": rate})
ax.text(1, 57, "Branch 1: Waymo pre-crash state          Branch 2: injury outcome          End-state frequency",
        fontsize=7.8, color=INK2)
ax.set_title("Event tree for Waymo rider-only crashes (posterior medians)")
fig.tight_layout(); fig.savefig(OUT / "figK_event_tree.png", bbox_inches="tight"); plt.close(fig)
pd.DataFrame(end).to_csv(OUT / "table21_event_tree_endstates.csv", index=False)
print("done")
for f in ["table14_posterior_percentiles", "table15_reduction_percentiles", "table16_exceedance", "table17_overlap",
          "table18_benchmark_variants", "table19_freq_vs_bayes", "table20_power_miles", "table21_event_tree_endstates"]:
    print("\n#", f); print(pd.read_csv(OUT / f"{f}.csv").round(4).to_string(index=False))
