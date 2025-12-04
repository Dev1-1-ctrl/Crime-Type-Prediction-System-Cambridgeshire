import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# CONFIG
INPUT_CSV = "crime_type_features_WITH_SYNTHETIC_CONTEXT.csv"
TEST_YEAR = 2025
USE_SMOTE = True   # For better class balance, using oversampling.

# LOAD
df = pd.read_csv(INPUT_CSV)
if "YearMonth" not in df.columns:
    df["YearMonth"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month_num"], day=1))
else:
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])

df["Crime type"] = df["Crime type"].astype(str).str.strip()
df["LSOA name"] = df["LSOA name"].astype(str).str.strip()

# GROUP CLASSES
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

# CREATE HISTORY FEATURES
monthly = (
    df.groupby(["LSOA name", "Year", "Month_num",
               "Crime_group"], as_index=False)
    .size()
    .rename(columns={"size": "cnt"})
)
monthly["YearMonth"] = pd.to_datetime(
    dict(year=monthly["Year"], month=monthly["Month_num"], day=1))

wide = (
    monthly.pivot_table(index=["LSOA name", "YearMonth"],
                        columns="Crime_group", values="cnt", fill_value=0)
    .reset_index()
)
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    if g not in wide.columns:
        wide[g] = 0

wide = wide.sort_values(["LSOA name", "YearMonth"]).copy()
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    wide[f"lag1_{g}"] = wide.groupby("LSOA name")[g].shift(1).fillna(0)

lags = wide[["LSOA name", "YearMonth"] +
            [f"lag1_{g}" for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]]]
df = df.merge(lags, on=["LSOA name", "YearMonth"], how="left")
for c in [f"lag1_{g}" for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]]:
    df[c] = df[c].fillna(0)

#  ENCODE
lsoa_enc = LabelEncoder()
df["LSOA_encoded"] = lsoa_enc.fit_transform(df["LSOA name"])

tgt_enc = LabelEncoder()
df["y_group"] = tgt_enc.fit_transform(df["Crime_group"])

# FEATURES
base_feats = ["Year", "Month_num", "Season",
              "Latitude", "Longitude", "LSOA_encoded"]
lag_feats = [f"lag1_{g}" for g in [
    "Anti-social", "Violence", "Theft", "Drugs", "Other"]]
synth_feats = [c for c in [
    "last_month_total_crimes", "unemployment_rate", "median_income", "education_index",
    "population_density", "age_15_24_share", "urban_flag",
    "cctv_density", "street_lighting_score",
    "avg_temp_c", "avg_rain_mm",
    "events_index", "public_event_flag", "bank_holiday_flag",
    "policing_presence_index", "stop_search_rate_per_1k"
] if c in df.columns]

X_cols = base_feats + lag_feats + synth_feats
y_col = "y_group"
df = df.dropna(subset=X_cols + [y_col]).copy()

# Time split
train_mask = df["Year"] < TEST_YEAR
test_mask = df["Year"] == TEST_YEAR
X_train = df.loc[train_mask, X_cols].reset_index(drop=True)
y_train = df.loc[train_mask, y_col].reset_index(drop=True)
X_test = df.loc[test_mask, X_cols].reset_index(drop=True)
y_test = df.loc[test_mask, y_col].reset_index(drop=True)

print(f"Train: {X_train.shape},  Test: {X_test.shape}")
print("Classes:", list(tgt_enc.classes_))

# SMOTE
if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(" SMOTE applied. New train size:", X_train.shape)
    except Exception as e:
        print("SMOTE not applied:", e)

# CLASS WEIGHTS
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, weights))
print("Class Weights:", class_weights_dict)

# TRAIN MODEL
model = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

sample_weights = np.array([class_weights_dict[y] for y in y_train])
model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
top3 = top_k_accuracy_score(y_test, model.predict_proba(X_test), k=3)

print(f"\n Grouped Accuracy: {acc:.2%}")
print(f" Grouped Macro F1: {f1m:.3f}")
print(f" Top-3 Accuracy: {top3:.2%}\n")

print("Classification Report (Grouped):")
print(classification_report(y_test, y_pred,
      target_names=tgt_enc.classes_, digits=3))

# Saving
joblib.dump(model, "crime_type_xgb_GROUPED_BALANCED_model.joblib")
joblib.dump(lsoa_enc, "lsoa_encoder.joblib")
joblib.dump(tgt_enc, "crime_group_encoder.joblib")
pd.Series(X_cols).to_csv("features_used_GROUPED_BALANCED.txt", index=False)
print("\n Saved: crime_type_xgb_GROUPED_BALANCED_model.joblib, encoders, and feature list.")
