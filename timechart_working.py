import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

FRED_API_KEY = "3707355de3032aa9b43716f690e0cf29"
START_DATE = "1950-01-01"


def fetch_fred_series(series_id, api_key, start_date="1947-01-01"):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    return df


# -------------------------------------------------------------------
# 1. Download data and construct annual profit rate
# -------------------------------------------------------------------

profits_q = fetch_fred_series("CP", FRED_API_KEY, START_DATE)
capital_a = fetch_fred_series("K1NTOTL1ES000", FRED_API_KEY, START_DATE)

profits_q["year"] = profits_q["date"].dt.year
capital_a["year"] = capital_a["date"].dt.year

profits_a = profits_q.groupby("year")["value"].mean().rename("profits_bil").to_frame()
capital_a_year = capital_a.groupby("year")["value"].mean().rename("capital_mil").to_frame()

df = profits_a.join(capital_a_year, how="inner")
df = df.loc[df.index >= 1950]

df["capital_bil"] = df["capital_mil"] / 1000.0
df["profit_rate_pct"] = (df["profits_bil"] / df["capital_bil"]) * 100.0
df = df.dropna(subset=["profit_rate_pct"])

# create year-end date for plotting
df["date"] = pd.to_datetime(df.index.astype(str) + "-12-31")

# -------------------------------------------------------------------
# 2. Define periods (regimes)
# -------------------------------------------------------------------

regimes = [
    {
        "label": "Golden age of accumulation",
        "start": "1950-01-01",
        "end":   "1969-12-31",
        "color": "#c6dbef",
    },
    {
        "label": "Stagflation (1970s)",
        "start": "1970-01-01",
        "end":   "1979-12-31",
        "color": "#fdd0a2",
    },
    {
        "label": "Neoliberal era (1980–2007)",
        "start": "1980-01-01",
        "end":   "2007-12-31",
        "color": "#c7e9c0",
    },
    {
        "label": "Post-2007 \"new normal\"",
        "start": "2008-01-01",
        "end":   df["date"].max().strftime("%Y-%m-%d"),
        "color": "#e0e0e0",
    },
]

# -------------------------------------------------------------------
# 3. Plot line + shaded periods
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

# regime shading
patches = []
for reg in regimes:
    start = pd.to_datetime(reg["start"])
    end = pd.to_datetime(reg["end"])
    ax.axvspan(start, end, color=reg["color"], alpha=0.4, zorder=0)
    patches.append(mpatches.Patch(color=reg["color"], alpha=0.4, label=reg["label"]))

# profit-rate line
line = ax.plot(
    df["date"],
    df["profit_rate_pct"],
    color="black",
    linewidth=2,
    label="Profit rate"
)[0]

ax.set_title("U.S. Corporate Rate of Profit and Economic Regimes, 1950–Present")
ax.set_xlabel("Year")
ax.set_ylabel("Rate of profit (% of net private nonresidential fixed assets)")

ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="y", alpha=0.3)

ymin, ymax = df["profit_rate_pct"].min(), df["profit_rate_pct"].max()
ax.set_ylim(ymin * 0.9, ymax * 1.1)

# legend: line + patches
handles = [line] + patches
ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8)

fig.tight_layout()

# save + show (Spyder will also show in Plots pane)
output_path = "profit_rate_timeline.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to {output_path}")

plt.show()
