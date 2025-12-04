import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# CONFIG
# "crime_type_features.csv"
INPUT_CSV = "crime_type_features_WITH_SYNTHETIC_CONTEXT.csv"
TEST_YEAR = 2025

# Loading
df = pd.read_csv(INPUT_CSV)
if "YearMonth" not in df.columns:
    df["YearMonth"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month_num"], day=1))
else:
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])

df["Crime type"] = df["Crime type"].astype(str).str.strip()
df["LSOA name"] = df["LSOA name"].astype(str).str.strip()

# GROUPS
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

# LAG FEATURES
monthly = (
    df.groupby(["LSOA name", "Year", "Month_num",
               "Crime_group"], as_index=False)
    .size().rename(columns={"size": "cnt"})
)
monthly["YearMonth"] = pd.to_datetime(
    dict(year=monthly["Year"], month=monthly["Month_num"], day=1))

wide = monthly.pivot_table(index=["LSOA name", "YearMonth"],
                           columns="Crime_group", values="cnt", fill_value=0).reset_index()
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    if g not in wide.columns:
        wide[g] = 0

wide = wide.sort_values(["LSOA name", "YearMonth"]).copy()
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    wide[f"lag1_{g}"] = wide.groupby("LSOA name")[g].shift(1).fillna(0)

lag_cols = ["LSOA name", "YearMonth"] + \
    [f"lag1_{g}" for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]]
lags = wide[lag_cols].copy()

df = df.merge(lags, on=["LSOA name", "YearMonth"], how="left")
lag_feat_cols = [c for c in df.columns if c.startswith("lag1_")]
df[lag_feat_cols] = df[lag_feat_cols].fillna(0)

# Encode
lsoa_enc = LabelEncoder()
df["LSOA_encoded"] = lsoa_enc.fit_transform(df["LSOA name"])

tgt_enc = LabelEncoder()
df["y_group"] = tgt_enc.fit_transform(df["Crime_group"])

# Features
base_feats = ["Year", "Month_num", "Season",
              "Latitude", "Longitude", "LSOA_encoded"]
synth_feats = [c for c in [
    "last_month_total_crimes",
    "unemployment_rate", "median_income", "education_index",
    "population_density", "age_15_24_share", "urban_flag",
    "cctv_density", "street_lighting_score",
    "avg_temp_c", "avg_rain_mm",
    "events_index", "public_event_flag", "bank_holiday_flag",
    "policing_presence_index", "stop_search_rate_per_1k"
] if c in df.columns]

X_cols = base_feats + lag_feat_cols + synth_feats
y_col = "y_group"

df = df.dropna(subset=X_cols + [y_col]).copy()

# Time split
train_mask = df["Year"] < TEST_YEAR
test_mask = df["Year"] == TEST_YEAR

X_train = df.loc[train_mask, X_cols].reset_index(drop=True)
y_train = df.loc[train_mask, y_col].reset_index(drop=True)
X_test = df.loc[test_mask,  X_cols].reset_index(drop=True)
y_test = df.loc[test_mask,  y_col].reset_index(drop=True)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print("Grouped classes:", list(tgt_enc.classes_))

# SMOTE (balance 5 groups on TRAIN ONLY)
print("Before SMOTE:", y_train.value_counts().to_dict())
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print("After SMOTE:", y_train_bal.value_counts().to_dict())

# TRAIN XGBOOST
model = XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)

# Evaluate
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
top3 = top_k_accuracy_score(y_test, proba, k=3)

print(f"\n Grouped Accuracy (SMOTE): {acc:.2%}")
print(f" Grouped Macro F1 (SMOTE): {f1m:.3f}")
print(f" Top‑3 Accuracy: {top3:.2%}\n")

print("Classification Report (Grouped, SMOTE):")
print(classification_report(y_test, y_pred,
      target_names=tgt_enc.classes_, digits=3))

# Saving
joblib.dump(model, "crime_type_xgb_GROUPED_SMOTE_model.joblib")
joblib.dump(lsoa_enc, "lsoa_encoder.joblib")           # ok to overwrite
joblib.dump(tgt_enc,  "crime_group_encoder.joblib")
pd.Series(X_cols).to_csv("features_used_GROUPED_SMOTE.txt", index=False)
print("\n Saved: crime_type_xgb_GROUPED_SMOTE_model.joblib, crime_group_encoder.joblib, features_used_GROUPED_SMOTE.txt")
