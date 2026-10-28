# Data

Raw data files are not stored in this repository. Download them into this folder before running the pipeline.

## NHTSA Standing General Order 2021-01: ADS incident reports

Source: https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting#data

| Save as | File on the NHTSA page |
|---|---|
| `data/SGO-2021-01_Incident_Reports_ADS.csv` | Current ADS incident reports (`SGO-2021-01_Incident_Reports_ADS.csv`) |
| `data/ADS_archive.csv` | Archive 2021–2025 ADS incident reports (same file name; rename it) |

The analysis in `results/` used the files available in October 2026: 2,295 archive rows and 1,595 current rows.

## Waymo Safety Impact hub

Source: https://waymo.com/safety/impact/ (download section, data through June 2026)

| Save as | File |
|---|---|
| `data/waymo/CSV2.csv` | CSV2 - Crashes with SGO ID and Group Membership 202009-202606-2022benchmark.csv |
| `data/waymo/CSV3.csv` | CSV3 - Collision Counts and Comparison to Benchmarks 202009-202606-2022benchmark.csv |

Then run `python prepare_waymo_data.py data/waymo/CSV2.csv data/waymo/CSV3.csv data` to create the compact inputs used by the comparison scripts.
