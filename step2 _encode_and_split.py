# =========================
# Step 2 — Encode & Split
# (train=2017–2024, test=2025)
# =========================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# 0) CONFIG
# or "crime_type_features.csv"
INPUT_CSV = "crime_type_features_WITH_SYNTHETIC_CONTEXT.csv"
# set True to oversample minority classes (requires: pip install imbalanced-learn)
APPLY_SMOTE = False

# 1) LOAD
df = pd.read_csv(INPUT_CSV)

# 2) DEDUPE
dups = df.duplicated().sum()
if dups:
    df = df.drop_duplicates().copy()
print(f"Duplicates removed: {dups}")

# 3) BASIC MISSING CHECKS
# Essential columns that must be present & non-null
essential = ["Year", "Month_num", "Season",
             "Latitude", "Longitude", "LSOA name", "Crime type"]
missing_cols = [c for c in essential if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"Missing essential columns in {INPUT_CSV}: {missing_cols}")

before = len(df)
df = df.dropna(subset=essential).copy()
print(f"Rows dropped for missing essentials: {before - len(df)}")

# 4) TEXT CLEANUP
df["LSOA name"] = df["LSOA name"].astype(str).str.strip()
df["Crime type"] = df["Crime type"].astype(str).str.strip()

# 5) ENCODE CATEGORICALS
lsoa_enc = LabelEncoder()
crime_enc = LabelEncoder()
df["LSOA_encoded"] = lsoa_enc.fit_transform(df["LSOA name"])
df["Crime_type_encoded"] = crime_enc.fit_transform(df["Crime type"])

# 6) SELECT FEATURES
# Base features always used:
base_feats = ["Year", "Month_num", "Season",
              "Latitude", "Longitude", "LSOA_encoded"]

# If synthetic columns exist, include them automatically
synthetic_feats = [
    "last_month_total_crimes",
    "unemployment_rate", "median_income", "education_index",
    "population_density", "age_15_24_share", "urban_flag",
    "cctv_density", "street_lighting_score",
    "avg_temp_c", "avg_rain_mm",
    "events_index", "public_event_flag", "bank_holiday_flag",
    "policing_presence_index", "stop_search_rate_per_1k"
]
present_synth = [c for c in synthetic_feats if c in df.columns]

X_cols = base_feats + present_synth
y_col = "Crime_type_encoded"

print(f"Features used ({len(X_cols)}): {X_cols}")

# 7) TIME-AWARE SPLIT (Option A)
if 2025 not in set(df["Year"].unique()):
    raise ValueError(
        "Year 2025 not found. Ensure Step 1 included 2025, or adjust the split year.")

train_mask = df["Year"] < 2025
test_mask = df["Year"] == 2025

X_train = df.loc[train_mask, X_cols].reset_index(drop=True)
y_train = df.loc[train_mask, y_col].reset_index(drop=True)
X_test = df.loc[test_mask,  X_cols].reset_index(drop=True)
y_test = df.loc[test_mask,  y_col].reset_index(drop=True)

print(f"✅ Split: Train={len(X_train)} rows | Test(2025)={len(X_test)} rows")
print(
    f"Classes: {len(crime_enc.classes_)} → {list(crime_enc.classes_)[:5]} ...")

# 8) (OPTIONAL) BALANCE CLASSES WITH SMOTE
if APPLY_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"SMOTE applied → New Train size: {len(X_train)}")
    except Exception as e:
        print("⚠️ Could not apply SMOTE. Install with: pip install imbalanced-learn")
        print("Error:", e)

# 9) SAVE ARTIFACTS
# Full encoded dataset (for reference / reuse)
df.to_csv("crime_type_features_encoded.csv", index=False)

# Train/Test splits
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Encoders to decode predictions later
joblib.dump(lsoa_enc, "lsoa_encoder.joblib")
joblib.dump(crime_enc, "crime_type_encoder.joblib")

print("💾 Saved: crime_type_features_encoded.csv, X_train/y_train, X_test/y_test, lsoa_encoder.joblib, crime_type_encoder.joblib")
