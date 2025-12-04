import pandas as pd
import joblib

# CONFIG
MODEL_FILE = "crime_type_xgb_GROUPED_model.joblib"   # RandomForest model
LSOA_ENC = "lsoa_encoder.joblib"
GRP_ENC = "crime_group_encoder.joblib"
INPUT_CSV = "crime_type_features_WITH_SYNTHETIC_CONTEXT.csv"
PREDICT_YEAR = 2025

print("🔹 Loading model and data...")
model = joblib.load(MODEL_FILE)
lsoa_enc = joblib.load(LSOA_ENC)
grp_enc = joblib.load(GRP_ENC)

# Load the dataset
df = pd.read_csv(INPUT_CSV)
print(f"Loaded dataset with {len(df):,} rows")

# Cleaning text columns
df["LSOA name"] = df["LSOA name"].astype(str).str.strip()

# Encode LSOA names
if "LSOA_encoded" not in df.columns:
    df["LSOA_encoded"] = lsoa_enc.transform(df["LSOA name"])

# Select rows for chosen prediction year
future_mask = (df["Year"] == PREDICT_YEAR)
df_future = df.loc[future_mask].copy()
print(f"Predicting for {len(df_future):,} rows (Year={PREDICT_YEAR})...")

# Define feature list
features = [
    "Year", "Month_num", "Season", "Latitude", "Longitude", "LSOA_encoded",
    "lag1_Anti-social", "lag1_Violence", "lag1_Theft", "lag1_Drugs", "lag1_Other",
    "last_month_total_crimes", "unemployment_rate", "median_income",
    "education_index", "population_density", "age_15_24_share", "urban_flag",
    "cctv_density", "street_lighting_score", "avg_temp_c", "avg_rain_mm",
    "events_index", "public_event_flag", "bank_holiday_flag",
    "policing_presence_index", "stop_search_rate_per_1k"
]

# Create missing lag columns
for lag_col in ["lag1_Anti-social", "lag1_Violence", "lag1_Theft", "lag1_Drugs", "lag1_Other"]:
    if lag_col not in df_future.columns:
        df_future[lag_col] = 0

# Prepare features for prediction
X_future = df_future[features].apply(pd.to_numeric, errors="coerce").fillna(0)

# Predict with trained model
print("🔹 Predicting...")
y_pred = model.predict(X_future)

# Figure out what text labels the guesses mean
df_future["Predicted_group_encoded"] = y_pred
df_future["top_group"] = grp_enc.inverse_transform(y_pred)

# Add up the forecasts for each LSOA
# Average latitude/longitude (numeric) and most common predicted crime group (text)
geo_avg = df_future.groupby("LSOA name")[["Latitude", "Longitude"]].mean()
top_groups = df_future.groupby(
    "LSOA name")["top_group"].agg(lambda x: x.mode().iat[0])
predicted_per_lsoa = geo_avg.join(top_groups).reset_index()

# Saving predictions to CSV
OUTPUT_FILE = f"predicted_crime_{PREDICT_YEAR}_per_lsoa.csv"
predicted_per_lsoa.to_csv(OUTPUT_FILE, index=False)

print(f" Saved → {OUTPUT_FILE}")
print("Done! You can now use this file in step6_maps_and_charts.py for visualization.")
