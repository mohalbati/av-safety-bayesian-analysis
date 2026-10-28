"""Compare this analysis' Waymo rider-only crash set with Waymo's published crash list (Safety Impact CSV2).
Usage: python compare_waymo.py <waymo_crash_list_compact.txt> <results/sgo_ads_clean.csv> <out_dir>
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
CSV2, CLEAN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
w = pd.read_csv(CSV2, sep="|", skiprows=1, header=None, keep_default_na=False,
                names=["id", "ver", "ym", "county", "road", "ctype", "bits"], dtype=str)
short = ["w_in_transport", "w_dv_lt1", "w_police", "w_any_inj", "w_airbag_any", "w_airbag_ego", "w_serious", "w_fatal"]
for i, s in enumerate(short): w[s] = w.bits.str[i].astype(int)
w["has_sgo_id"] = ~w["id"].isin(["N/A", "#N/A", ""])
METRO = {"Maricopa": "Phoenix", "San Francisco": "SF Bay Area", "San Mateo": "SF Bay Area", "Santa Clara": "SF Bay Area",
         "Los Angeles": "Los Angeles", "Travis": "Austin", "Fulton": "Atlanta", "DeKalb": "Atlanta"}
w["metro"] = w["county"].map(METRO).fillna("Other")
w.to_csv(OUT / "waymo_csv2_parsed.csv", index=False)

d = pd.read_csv(CLEAN, low_memory=False); d["date"] = pd.to_datetime(d["date"])
wy = d[(d.entity == "Waymo") & (d.date <= "2026-06-01")]
st = wy.State.fillna("").str.strip(); city = wy.City.fillna("").str.strip().str.lower()
TXGA = ["austin", "atlanta", "decatur", "sandy springs", "east point", "college park", "brookhaven", "round rock",
        "pflugerville", "del valle", "sunset valley", "west lake hills", "rollingwood"]
ia = st.isin(["AZ", "CA"]) | (st.isin(["TX", "GA"]) & city.isin(TXGA))
ours = wy[ia & ~wy.test_driver & ~wy.ads_not_engaged & ~wy.manual_mode]

ws = w[w.has_sgo_id & (w.metro != "Other")].sort_values("ver").drop_duplicates("id", keep="last")
A, B = set(ours["Report ID"]), set(ws["id"])
rows = [("Waymo list, all rows", len(w)), ("  without an SGO ID (pre-SGO, Sep 2020 - Jun 2021)", int((~w.has_sgo_id).sum())),
        ("  outside the 5 benchmarked metros", int((w.has_sgo_id & (w.metro == "Other")).sum())),
        ("Waymo list, SGO-era, 5 metros", len(B)), ("This analysis", len(A)), ("In both", len(A & B)),
        ("Only in this analysis", len(A - B)), ("Only in Waymo list", len(B - A))]
ov = pd.DataFrame(rows, columns=["set", "vehicles"]); ov.to_csv(OUT / "table11_set_overlap.csv", index=False)

# outcome agreement on the shared reports
m = ours.merge(ws, left_on="Report ID", right_on="id", how="outer", indicator=True)
agree = []
for ocol, wcol, lab in [("inj_any", "w_any_inj", "Any injury"), ("airbag_any", "w_airbag_any", "Airbag (any vehicle)"),
                        ("inj_serious", "w_serious", "Serious or worse")]:
    o = m[ocol].fillna(False).astype(bool); t = m[wcol].fillna(0).astype(int).astype(bool)
    agree.append({"outcome": lab, "ours": int(o.sum()), "Waymo (SGO-era, 5 metros)": int(t.sum()),
                  "Waymo (incl. pre-SGO)": int(w[(w.metro != "Other")][wcol].sum()),
                  "both flag": int((o & t).sum()), "only ours": int((o & ~t).sum()), "only Waymo": int((~o & t).sum())})
ag = pd.DataFrame(agree); ag.to_csv(OUT / "table12_outcome_agreement.csv", index=False)

# why do sets differ? characterise the only-ours / only-Waymo reports
only_ours = ours[ours["Report ID"].isin(A - B)]
only_w = ws[ws["id"].isin(B - A)]
why = {"only_ours_by_SV_movement": only_ours["SV Pre-Crash Movement"].value_counts().head(6).to_dict(),
       "only_ours_crash_with": only_ours["Crash With"].value_counts().head(6).to_dict(),
       "only_waymo_crash_type": only_w["ctype"].value_counts().head(6).to_dict(),
       "only_waymo_in_our_clean_data": int(only_w["id"].isin(d["Report ID"]).sum())}
# for only-Waymo IDs present in our data, what excluded them?
ex = d[d["Report ID"].isin(only_w["id"])]
why["only_waymo_excluded_by"] = {"test_driver": int(ex.test_driver.sum()), "ads_not_engaged": int(ex.ads_not_engaged.sum()),
                                 "manual_mode": int(ex.manual_mode.sum()),
                                 "date_after_cutoff": int((pd.to_datetime(ex.date) > "2026-06-01").sum())}
# only-ours: crash partner fixed/parked objects (Waymo treats parked vehicles as fixed objects, excludes non-contact)
# outcome disagreements: what does the SGO severity field say for reports Waymo flags but we do not?
sh = ours.merge(ws, left_on="Report ID", right_on="id", how="inner")
why["waymo_injury_not_ours_sev"] = sh[(sh.w_any_inj == 1) & ~sh.inj_any]["Highest Injury Severity Alleged"].value_counts().to_dict()
why["ours_serious_not_waymo_sev"] = sh[(sh.w_serious == 0) & sh.inj_serious]["Highest Injury Severity Alleged"].value_counts().to_dict()
why["waymo_airbag_not_ours_field"] = sh[(sh.w_airbag_any == 1) & ~sh.airbag_any]["Report ID"].tolist()
pd.Series({k: str(v) for k, v in why.items()}).to_csv(OUT / "table13_set_difference_notes.csv")
print(ov.to_string(index=False)); print(ag.to_string(index=False))
for k, v in why.items(): print(k, v)
