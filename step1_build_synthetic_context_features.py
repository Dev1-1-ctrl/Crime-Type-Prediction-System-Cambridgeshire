import pandas as pd
import numpy as np

# ------------------------------------------------------
# 1) Load your Step 1 file (incident-level, one row = one crime)
# ------------------------------------------------------
df = pd.read_csv("crime_type_features.csv")

# Make a YearMonth key
df["YearMonth"] = pd.to_datetime(
    dict(year=df["Year"], month=df["Month_num"], day=1))

# ------------------------------------------------------
# 2) Build LSOA-month table (one row per LSOA x month)
# ------------------------------------------------------
monthly = (
    df.groupby(["LSOA name", "Year", "Month_num"], as_index=False)
      .size()
      .rename(columns={"size": "total_crimes"})
)
monthly["YearMonth"] = pd.to_datetime(
    dict(year=monthly["Year"], month=monthly["Month_num"], day=1))

# Add a lagged real feature: last month's total crimes in this LSOA
monthly = monthly.sort_values(["LSOA name", "YearMonth"])
monthly["last_month_total_crimes"] = (
    monthly.groupby("LSOA name")["total_crimes"].shift(1).fillna(0)
)

# ------------------------------------------------------
# 3) Create synthetic (fake) context features at LSOA level
#     - deterministic by LSOA (so it’s stable across months)
#     - gentle yearly trends + seasonal/weather where relevant
# ------------------------------------------------------
rng = np.random.RandomState(42)

# Stable per-LSOA baselines
lsoas = monthly["LSOA name"].unique()
lsoa_baselines = pd.DataFrame({
    "LSOA name": lsoas,
    # Socioeconomic baselines
    # 3%–12%
    "unemployment_base": rng.uniform(0.03, 0.12, size=len(lsoas)),
    # £20k–£45k
    "median_income_base": rng.uniform(20000, 45000, size=len(lsoas)),
    "education_index_base": rng.uniform(0.4, 0.85, size=len(lsoas)),      # 0–1
    # Demographics
    # ppl/km2
    "population_density_base": rng.uniform(200, 7000, size=len(lsoas)),
    # 8–25%
    "age_15_24_share_base": rng.uniform(0.08, 0.25, size=len(lsoas)),
})

# Urban flag from density
lsoa_baselines["urban_flag"] = (
    lsoa_baselines["population_density_base"] > 1200).astype(int)

# CCTV & lighting depend on urban-ness
lsoa_baselines["cctv_density_base"] = (lsoa_baselines["urban_flag"] * rng.uniform(5, 25, len(lsoas))
                                       + (1 - lsoa_baselines["urban_flag"]) * rng.uniform(0, 8, len(lsoas)))
lsoa_baselines["street_lighting_base"] = (lsoa_baselines["urban_flag"] * rng.uniform(3.0, 4.8, len(lsoas))
                                          + (1 - lsoa_baselines["urban_flag"]) * rng.uniform(2.0, 4.2, len(lsoas)))

# Month-wise UK-ish weather profiles
month_temp = {1: 4, 2: 5, 3: 7, 4: 9, 5: 13, 6: 16,
              7: 18, 8: 18, 9: 15, 10: 11, 11: 7, 12: 5}
month_rain = {1: 55, 2: 45, 3: 45, 4: 45, 5: 50,
              6: 55, 7: 55, 8: 60, 9: 55, 10: 65, 11: 60, 12: 60}

# Merge baselines to monthly
monthly = monthly.merge(lsoa_baselines, on="LSOA name", how="left")

# Gentle yearly trends + noise
year0 = monthly["Year"].min()
year_delta = (monthly["Year"] - year0)

# Socioeconomic
monthly["unemployment_rate"] = (monthly["unemployment_base"]
                                + 0.001 * year_delta
                                + rng.normal(0, 0.002, len(monthly))) \
    .clip(0.02, 0.15)

monthly["median_income"] = (monthly["median_income_base"]
                            + 350 * year_delta
                            + rng.normal(0, 500, len(monthly))) \
    .clip(18000, 60000)

monthly["education_index"] = (monthly["education_index_base"]
                              + 0.002 * year_delta
                              + rng.normal(0, 0.01, len(monthly))) \
    .clip(0.3, 0.95)

# Demographics (mostly stable)
monthly["population_density"] = (monthly["population_density_base"]
                                 + rng.normal(0, 30, len(monthly))) \
    .clip(100, 10000)
monthly["age_15_24_share"] = (monthly["age_15_24_share_base"]
                              + rng.normal(0, 0.005, len(monthly))) \
    .clip(0.05, 0.35)
monthly["urban_flag"] = monthly["urban_flag"]

# Environmental
monthly["cctv_density"] = (monthly["cctv_density_base"]
                           + rng.normal(0, 1.0, len(monthly))) \
    .clip(0, 40)
monthly["street_lighting_score"] = (monthly["street_lighting_base"]
                                    + rng.normal(0, 0.2, len(monthly))) \
    .clip(1.0, 5.0)

monthly["avg_temp_c"] = monthly["Month_num"].map(
    month_temp) + rng.normal(0, 1.0, len(monthly))
monthly["avg_rain_mm"] = monthly["Month_num"].map(
    month_rain) + rng.normal(0, 5.0, len(monthly))

# Events (more in summer months)
summer = monthly["Month_num"].isin([5, 6, 7, 8, 9]).astype(int)
monthly["events_index"] = rng.poisson(
    0.6 + 0.8 * summer, size=len(monthly)).astype(float)
monthly["public_event_flag"] = (monthly["events_index"] > 0).astype(int)

# Simple UK bank-holiday-ish months (approx.)
monthly["bank_holiday_flag"] = monthly["Month_num"].isin(
    [1, 4, 5, 8, 12]).astype(int)

# Policing presence tied to last month's crimes + noise
monthly["policing_presence_index"] = (0.5 * np.log1p(monthly["last_month_total_crimes"])
                                      + 0.3 * monthly["urban_flag"]
                                      + rng.normal(0, 0.3, len(monthly))) \
    .clip(0, None)

# Stop & search rate per 1k population: depends on presence + urban
monthly["stop_search_rate_per_1k"] = (monthly["policing_presence_index"] * (2 + 3*monthly["urban_flag"])
                                      + rng.normal(0, 0.5, len(monthly))) \
    .clip(0, None)

# Keep the synthetic columns to merge back
synthetic_cols = [
    "LSOA name", "Year", "Month_num", "YearMonth",
    "last_month_total_crimes",
    "unemployment_rate", "median_income", "education_index",
    "population_density", "age_15_24_share", "urban_flag",
    "cctv_density", "street_lighting_score",
    "avg_temp_c", "avg_rain_mm",
    "events_index", "public_event_flag", "bank_holiday_flag",
    "policing_presence_index", "stop_search_rate_per_1k"
]
monthly_synth = monthly[synthetic_cols].copy()

# ------------------------------------------------------
# 4) Merge synthetic monthly features back onto each crime row
# ------------------------------------------------------
df_aug = df.merge(
    monthly_synth,
    on=["LSOA name", "Year", "Month_num", "YearMonth"],
    how="left"
)

# ------------------------------------------------------
# 5) Save augmented dataset
# ------------------------------------------------------
df_aug.to_csv("crime_type_features_WITH_SYNTHETIC_CONTEXT.csv", index=False)
print("✅ Saved → crime_type_features_WITH_SYNTHETIC_CONTEXT.csv")
print("New columns added:", [
      c for c in df_aug.columns if c not in df.columns][:10], "...")
print("Rows:", len(df_aug), " | Example row:\n", df_aug.head(2))
