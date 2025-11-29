import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path(r"C:\Users\Ignacio\projects\ucla\fall25\econometrics\project3\data")

# -----------------------------
# Helpers for quarterly indexes
# -----------------------------
def load_quarterly_csv(fname):
    """
    Read a CSV where the first column is a quarterly index
    (e.g. '1963Q1' or a date), convert to PeriodIndex(Q-DEC).
    Works with the CSVs produced by data_grabber.py.
    """
    df = pd.read_csv(data_dir / fname, index_col=0)

    # Try '1963Q1' style first
    try:
        idx = pd.PeriodIndex(df.index.astype(str), freq="Q-DEC")
    except Exception:
        # Fallback: parse as datetime, then to quarter
        idx = pd.to_datetime(df.index).to_period("Q-DEC")

    df.index = idx
    return df.sort_index()

def parse_quarter_dates(df, col="Quarter"):
    """
    For files where quarter is stored in a date column (e.g. '3/31/1962'),
    convert that column to PeriodIndex(Q-DEC).
    """
    dt = pd.to_datetime(df[col])
    df = df.drop(columns=[col])
    df.index = pd.DatetimeIndex(dt).to_period("Q-DEC")
    return df.sort_index()


# ==============================
# 1. R – recession indicator
# ==============================
R_df = load_quarterly_csv("R_quarterly.csv")        # column: R

# ==============================
# 2. CGDP – credit-to-GDP ratio (level)
# ==============================
CGDP_df = load_quarterly_csv("CGDP_quarterly.csv")  # column: CGDP

# ==============================
# 3. CPIR – inflation (from CPIAUCSL)
# ==============================
CPIR_df = load_quarterly_csv("CPIR_quarterly.csv")  # column: CPIR

# ==============================
# 4. INVR – investment growth
# ==============================
INVR_df = load_quarterly_csv("INVR_quarterly.csv")
INVR_df = INVR_df[["INVR"]]

# ==============================
# 5. HP – house price change (dHP_log)
# ==============================
HP_df = load_quarterly_csv("HP_quarterly.csv")      # HP_index, dHP_log
HP_df = HP_df.rename(columns={"dHP_log": "HP"})
HP_df = HP_df[["HP"]]   # keep only the change, not the level

# ==============================
# 6. IR – real interest rate
# ==============================
IR_df = load_quarterly_csv("IR_quarterly.csv")      # IR_nom, CPIR, IR_real
IR_df = IR_df.rename(columns={"IR_real": "IR"})
IR_df = IR_df[["IR"]]

# ==============================
# 7. ROP – profit rate
# ==============================
ROP_df = load_quarterly_csv("ROP_quarterly.csv")    # profits, capital, ROP
ROP_df = ROP_df[["ROP"]]

# ==============================
# 8. IROR – incremental rate of profit
# ==============================
IROR_df = load_quarterly_csv("IROR_quarterly.csv")  # has IROR, IROR_pct, etc.
IROR_df = IROR_df[["IROR"]]   # use the level

# ==============================
# 9. LSOI – labour share
# ==============================
LSOI_df = load_quarterly_csv("LSOI_quarterly.csv")  # column: LSOI

# ==============================
# 10. RGDP – real GDP growth (RGDP_growth)
# ==============================
RGDP_df = load_quarterly_csv("RGDP_quarterly.csv")  # RGDP, RGDP_growth
RGDP_df = RGDP_df.rename(columns={"RGDP_growth": "RGDP"})
RGDP_df = RGDP_df[["RGDP"]]

# ==============================
# 11. MA – PMI manufacturing (manual)
#      MA_quarterly.csv: Quarter (date), MA
# ==============================
MA_raw = pd.read_csv(data_dir / "MA_quarterly.csv")
MA_df = parse_quarter_dates(MA_raw, col="Quarter")
MA_df = MA_df.rename(columns={"MA": "MA"})

# ==============================
# 12. TOBQ – Tobin’s Q (manual Excel)
#      TOBQ_quarterly.xlsx: Quarter (date), TOBQ
# ==============================
TOBQ_raw = pd.read_excel(data_dir / "TOBQ_quarterly.xlsx")
TOBQ_df = parse_quarter_dates(TOBQ_raw, col="Quarter")
TOBQ_df = TOBQ_df.rename(columns={"TOBQ": "TOBQ"})

# ==============================
# 13. Align sample and merge
# ==============================
start_q = pd.Period("1963Q1", freq="Q-DEC")
end_q   = pd.Period("2018Q2", freq="Q-DEC")

dfs = [
    R_df,
    CGDP_df,
    CPIR_df,
    INVR_df,
    HP_df,
    IR_df,
    ROP_df,
    IROR_df,
    LSOI_df,
    RGDP_df,
    MA_df,
    TOBQ_df,
]

master = pd.concat(dfs, axis=1)

# Restrict to desired window
master = master.loc[start_q:end_q]

# Diagnostics
print("Date range:", master.index.min(), "to", master.index.max())
print("Rows:", len(master))
print(master.isna().sum())

# Save master
out_path = data_dir / "master_quarterly.csv"
master.to_csv(out_path)
print("Saved master to:", out_path)
