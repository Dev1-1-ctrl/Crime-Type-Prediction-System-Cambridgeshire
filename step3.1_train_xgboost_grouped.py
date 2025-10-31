# Step 3 — Boosted version: Grouped classes + history + XGBoost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score
import joblib

# ---------- CONFIG ----------
# or "crime_type_features.csv"
INPUT_CSV = "crime_type_features_WITH_SYNTHETIC_CONTEXT.csv"
TEST_YEAR = 2025
USE_SMOTE = False  # optional; set True if you want to oversample after grouping

# ---------- LOAD ----------
df = pd.read_csv(INPUT_CSV)
# Make YearMonth if missing
if "YearMonth" not in df.columns:
    df["YearMonth"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month_num"], day=1))
else:
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])

# Clean text
df["Crime type"] = df["Crime type"].astype(str).str.strip()
df["LSOA name"] = df["LSOA name"].astype(str).str.strip()

# ---------- 1) GROUP CLASSES (14 -> 5 buckets) ----------
crime_map = {
    "Anti-social behaviour": "Anti-social",
    "Violence and sexual offences": "Violence",
    "Bicycle theft": "Theft",
    "Burglary": "Theft",
    "Criminal damage and arson": "Violence",
    "Drugs": "Drugs",
    "Other crime": "Other",
    "Other theft": "Theft",
    "Possession of weapons": "Violence",
    "Public order": "Other",
    "Robbery": "Theft",
    "Shoplifting": "Theft",
    "Theft from the person": "Theft",
    "Vehicle crime": "Theft",
}
df["Crime_group"] = df["Crime type"].map(crime_map).fillna("Other")

# ---------- 2) HISTORY FEATURES (last month counts per LSOA per group) ----------
# Build monthly counts per LSOA x Crime_group
monthly = (
    df.groupby(["LSOA name", "Year", "Month_num",
               "Crime_group"], as_index=False)
    .size()
    .rename(columns={"size": "cnt"})
)
monthly["YearMonth"] = pd.to_datetime(
    dict(year=monthly["Year"], month=monthly["Month_num"], day=1))

# Pivot to wide: one column per group
wide = (
    monthly.pivot_table(index=["LSOA name", "YearMonth"],
                        columns="Crime_group", values="cnt", fill_value=0)
    .reset_index()
)
# Ensure all 5 groups exist as columns
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    if g not in wide.columns:
        wide[g] = 0

# Sort and create lag-1 features (previous month’s counts per group)
wide = wide.sort_values(["LSOA name", "YearMonth"]).copy()
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    wide[f"lag1_{g}"] = wide.groupby("LSOA name")[g].shift(1).fillna(0)

# Keep only the lag features to merge back
lag_cols = ["LSOA name", "YearMonth"] + \
    [f"lag1_{g}" for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]]
lags = wide[lag_cols].copy()

# Merge lag features onto each incident row
df = df.merge(lags, on=["LSOA name", "YearMonth"], how="left")
df[[c for c in df.columns if c.startswith("lag1_")]] = df[[
    c for c in df.columns if c.startswith("lag1_")]].fillna(0)

# ---------- 3) ENCODE LSOA + TARGET (grouped) ----------
lsoa_enc = LabelEncoder()
df["LSOA_encoded"] = lsoa_enc.fit_transform(df["LSOA name"])

tgt_enc = LabelEncoder()
df["y_group"] = tgt_enc.fit_transform(df["Crime_group"])

# ---------- 4) FEATURE MATRIX ----------
base_feats = ["Year", "Month_num", "Season",
              "Latitude", "Longitude", "LSOA_encoded"]
lag_feats = [f"lag1_{g}" for g in [
    "Anti-social", "Violence", "Theft", "Drugs", "Other"]]

# include synthetic context if present
synth_feats = [c for c in [
    "last_month_total_crimes",
    "unemployment_rate", "median_income", "education_index",
    "population_density", "age_15_24_share", "urban_flag",
    "cctv_density", "street_lighting_score",
    "avg_temp_c", "avg_rain_mm",
    "events_index", "public_event_flag", "bank_holiday_flag",
    "policing_presence_index", "stop_search_rate_per_1k"
] if c in df.columns]

X_cols = base_feats + lag_feats + synth_feats
y_col = "y_group"

# Drop any remaining NA in required columns
df = df.dropna(subset=X_cols + [y_col]).copy()

# ---------- 5) TIME-AWARE SPLIT ----------
train_mask = df["Year"] < TEST_YEAR
test_mask = df["Year"] == TEST_YEAR

X_train = df.loc[train_mask, X_cols].reset_index(drop=True)
y_train = df.loc[train_mask, y_col].reset_index(drop=True)
X_test = df.loc[test_mask,  X_cols].reset_index(drop=True)
y_test = df.loc[test_mask,  y_col].reset_index(drop=True)

print(f"Train: {X_train.shape},  Test: {X_test.shape}")
print("Classes:", list(tgt_enc.classes_))

# ---------- 6) (OPTIONAL) SMOTE AFTER GROUPING ----------
if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("SMOTE applied. New train size:", X_train.shape)
    except Exception as e:
        print("SMOTE not applied:", e)

# ---------- 7) TRAIN XGBOOST ----------
model = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------- 8) EVALUATE ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
top3 = top_k_accuracy_score(y_test, model.predict_proba(X_test), k=3)

print(f"\n✅ Grouped Accuracy: {acc:.2%}")
print(f"✅ Grouped Macro F1: {f1m:.3f}")
print(f"✅ Top‑3 Accuracy: {top3:.2%}")

# per-class report (readable names)
print("\nClassification Report (Grouped):")
print(classification_report(y_test, y_pred,
      target_names=tgt_enc.classes_, digits=3))

# ---------- 9) SAVE ARTIFACTS ----------
joblib.dump(model, "crime_type_xgb_GROUPED_model.joblib")
joblib.dump(lsoa_enc, "lsoa_encoder.joblib")            # re-saving ok
joblib.dump(tgt_enc,  "crime_group_encoder.joblib")
pd.Series(X_cols).to_csv("features_used_GROUPED.txt", index=False)
print("\n💾 Saved: crime_type_xgb_GROUPED_model.joblib, crime_group_encoder.joblib, features_used_GROUPED.txt")
