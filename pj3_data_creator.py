"""
data_grabber.py

Fetch all FRED-based quarterly series for the Alexiou–Trachanas replication
and save them as CSV files.

MA and TOBQ remain manual.
"""

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from fredapi import Fred

# ============================================================
# 0. General setup
# ============================================================

# Insert your real API key here
FRED_API_KEY = "3707355de3032aa9b43716f690e0cf29"

data_dir = Path(r"C:\Users\Ignacio\projects\ucla\fall25\econometrics\project3\data")
data_dir.mkdir(parents=True, exist_ok=True)

# Fetch a buffer window before and after the actual sample
START_DATE = "1961-01-01"     # gives 1961Q1–1962Q4 buffer for growth rates
END_DATE   = "2019-01-01"     # gives two quarters beyond 2018Q2

# Analysis sample still the same
START_Q = "1963Q1"
END_Q   = "2018Q2"

fred = Fred(api_key=FRED_API_KEY)


def fred_q(series_id, start=START_DATE, end=END_DATE):
    """
    Download a FRED series at quarterly frequency using FRED's own
    quarterly construction (frequency='q'), clean '.' to NaN, and
    return a Series with PeriodIndex(freq='Q-DEC') restricted to
    1963Q1–2018Q2.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
        "frequency": "q",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    obs = pd.DataFrame(r.json()["observations"])
    obs["date"] = pd.to_datetime(obs["date"])
    obs["value"] = pd.to_numeric(obs["value"].replace(".", np.nan))
    obs = obs.set_index("date")
    obs.index = obs.index.to_period("Q-DEC")
    s = obs["value"]
    s = s.loc[START_Q:END_Q]
    return s


def fetch_fred_series(series_id, start=START_DATE, end=END_DATE):
    """
    Download a FRED series at its native frequency (no frequency param),
    convert to PeriodIndex(freq='Q-DEC'), and restrict to 1963Q1–2018Q2.

    Used for series that are already quarterly in FRED (e.g. W326RC1Q027SBEA).
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    obs = resp.json()["observations"]

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan))
    df = df.set_index("date")
    df.index = df.index.to_period("Q-DEC")
    df = df.loc[START_Q:END_Q]

    return df


# ============================================================
# 1. R – Binary recession indicator (USRECQP)
# ============================================================

R = fred_q("USRECQP")                     # binary NBER recession
R_df = R.to_frame(name="R")
out_path = data_dir / "R_quarterly.csv"
R_df.to_csv(out_path)
print("Saved R to:", out_path)


# ============================================================
# 2. CGDP – Credit-to-GDP ratio (QUSPAM770A)
# ============================================================

CGDP = fred_q("QUSPAM770A")
CGDP_df = CGDP.to_frame(name="CGDP")
out_path = data_dir / "CGDP_quarterly.csv"
CGDP_df.to_csv(out_path)
print("Saved CGDP to:", out_path)


# ============================================================
# 3. CPIR – Inflation from CPIAUCSL
# ============================================================

# Step 1: monthly CPI
cpi_m = fred.get_series("CPIAUCSL")
cpi_m = cpi_m.loc[START_DATE:END_DATE]

# Step 2: convert to quarterly average
cpi_q = cpi_m.resample("Q").mean()
cpi_q.index = cpi_q.index.to_period("Q-DEC")

# Step 3: quarterly inflation (% change)
inflation_q = cpi_q.pct_change() * 100.0
inflation_q = inflation_q.rename("CPIR")
inflation_q = inflation_q.loc[START_Q:END_Q]

CPIR_df = inflation_q.to_frame()
out_path = data_dir / "CPIR_quarterly.csv"
CPIR_df.to_csv(out_path)
print("Saved CPIR (inflation) to:", out_path)


# ============================================================
# 4. INVR – Gross Fixed Capital Formation growth (NCBGCFQ027S)
# ============================================================

# start one year earlier so growth for 1963Q1 is defined
GFCF = fred_q("NCBGCFQ027S", start="1962-01-01", end=END_DATE)

df_INVR = GFCF.to_frame(name="GFCF")
df_INVR["INVR"] = 100.0 * (df_INVR["GFCF"] / df_INVR["GFCF"].shift(1) - 1)
df_INVR = df_INVR.loc[START_Q:END_Q]

out_path = data_dir / "INVR_quarterly.csv"
df_INVR.to_csv(out_path)
print("Saved INVR to:", out_path)


# ============================================================
# 5. HP – Residential House Prices Index (QUSN628BIS)
# ============================================================

HP = fred_q("QUSN628BIS")
hp_df = HP.to_frame(name="HP_index")

hp_df["dHP_log"] = 100.0 * (
    np.log(hp_df["HP_index"]) - np.log(hp_df["HP_index"].shift(1))
)
hp_df = hp_df.loc[START_Q:END_Q]

out_path = data_dir / "HP_quarterly.csv"
hp_df.to_csv(out_path)
print("Saved HP to:", out_path)


# ============================================================
# 6. IR – Real interest rate = TB3MS – CPIR
#     TB3MS quarterly via FRED's frequency='q', same as your original
# ============================================================

IR_nom = fred_q("TB3MS")      # FRED quarterly 3M T-bill
IR_nom = IR_nom.rename("IR_nom")

# Align with CPIR (already quarterly)
ir_df = pd.concat([IR_nom, inflation_q], axis=1)
ir_df.columns = ["IR_nom", "CPIR"]
ir_df = ir_df.loc[START_Q:END_Q]

ir_df["IR_real"] = ir_df["IR_nom"] - ir_df["CPIR"]

out_path = data_dir / "IR_quarterly.csv"
ir_df.to_csv(out_path)
print("Saved IR (nominal, CPIR, real) to:", out_path)



# ============================================================
# 7. ROP – Corporate Profits (using BEA series)
#     B471RC1Q027SBEA
# ============================================================

# Fetch BEA corporate profits (after tax, IVA, CCAdj)
rop_df = fetch_fred_series("B471RC1Q027SBEA")
rop_df = rop_df.rename(columns={"value": "ROP"})   # keep original naming convention

out_path = data_dir / "ROP_quarterly.csv"
rop_df.to_csv(out_path)
print("Saved ROP to:", out_path)



# ============================================================
# 8. IROR – Incremental rate of profit per unit of investment
#     profits_nom: W326RC1Q027SBEA
#     inv_nom: NCBGCFQ027S
#     gdpdef: GDPDEF
# ============================================================

profits_nom = fred_q("W326RC1Q027SBEA", start="1962-01-01", end=END_DATE)
inv_nom     = fred_q("NCBGCFQ027S",     start="1962-01-01", end=END_DATE)
gdpdef      = fred_q("GDPDEF",          start="1962-01-01", end=END_DATE)

df_IROR = pd.concat([profits_nom, inv_nom, gdpdef], axis=1)
df_IROR.columns = ["profits_nom", "inv_nom", "gdpdef"]

price = df_IROR["gdpdef"] / 100.0
df_IROR["profits_real"] = df_IROR["profits_nom"] / price
df_IROR["inv_real"]     = df_IROR["inv_nom"] / price

df_IROR["d_profits_real"] = df_IROR["profits_real"] - df_IROR["profits_real"].shift(1)
df_IROR["inv_real_lag"]   = df_IROR["inv_real"].shift(1)

df_IROR["IROR"]     = df_IROR["d_profits_real"] / df_IROR["inv_real_lag"]
df_IROR["IROR_pct"] = 100.0 * df_IROR["IROR"]

# first IROR defined at 1963Q2
df_IROR = df_IROR.loc["1963Q2":END_Q]

out_path = data_dir / "IROR_quarterly.csv"
df_IROR.to_csv(out_path)
print("Saved IROR to:", out_path)


# ============================================================
# 9. LSOI – Labour’s share of income (PRS85006173)
# ============================================================

LSOI = fred_q("PRS85006173")
LSOI_df = LSOI.to_frame(name="LSOI")
LSOI_df = LSOI_df.loc[START_Q:END_Q]

out_path = data_dir / "LSOI_quarterly.csv"
LSOI_df.to_csv(out_path)
print("Saved LSOI to:", out_path)


# ============================================================
# 10. RGDP – Real GDP and growth (GDPC1)
# ============================================================

RGDP = fred_q("GDPC1")
RGDP_df = RGDP.to_frame(name="RGDP")
RGDP_df["RGDP_growth"] = 100.0 * (RGDP_df["RGDP"] / RGDP_df["RGDP"].shift(1) - 1)
RGDP_df = RGDP_df.loc[START_Q:END_Q]

out_path = data_dir / "RGDP_quarterly.csv"
RGDP_df.to_csv(out_path)
print("Saved RGDP (level and growth) to:", out_path)


# ============================================================
# 11. MA and TOBQ – manual
# ============================================================

print("\nNOTE: MA_quarterly.csv and TOBQ_quarterly.xlsx are still manual inputs.")
