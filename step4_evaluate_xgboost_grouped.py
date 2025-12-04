import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
try:
    from sklearn.metrics import top_k_accuracy_score
except Exception:
    top_k_accuracy_score = None

# CONFIG
# "crime_type_features.csv"
INPUT_CSV = "crime_type_features_WITH_SYNTHETIC_CONTEXT.csv"
TEST_YEAR = 2025
MODEL_FILE = "crime_type_xgb_GROUPED_BALANCED_model.joblib"
LSOA_ENC = "lsoa_encoder.joblib"
GRP_ENC = "crime_group_encoder.joblib"
FEATS_FILE = "features_used_GROUPED_BALANCED.txt"


print("Loading data and artifacts...")
df = pd.read_csv(INPUT_CSV)
model: XGBClassifier = joblib.load(MODEL_FILE)
lsoa_enc = joblib.load(LSOA_ENC)
grp_enc = joblib.load(GRP_ENC)

# Pre-clean (same as Step 3)
if "YearMonth" not in df.columns:
    df["YearMonth"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month_num"], day=1))
else:
    df["YearMonth"] = pd.to_datetime(df["YearMonth"], errors="coerce")

df["Crime type"] = df["Crime type"].astype(str).str.strip()
df["LSOA name"] = df["LSOA name"].astype(str).str.strip()

# Same 14-5 group mapping
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

# Rebuild history features (lag1_*)
monthly = (
    df.groupby(["LSOA name", "Year", "Month_num",
               "Crime_group"], as_index=False)
    .size().rename(columns={"size": "cnt"})
)
monthly["YearMonth"] = pd.to_datetime(
    dict(year=monthly["Year"], month=monthly["Month_num"], day=1))
wide = (monthly.pivot_table(index=["LSOA name", "YearMonth"],
                            columns="Crime_group", values="cnt", fill_value=0)
        .reset_index())
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    if g not in wide.columns:
        wide[g] = 0
wide = wide.sort_values(["LSOA name", "YearMonth"]).copy()
for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]:
    wide[f"lag1_{g}"] = wide.groupby("LSOA name")[g].shift(1).fillna(0)

lag_cols = ["LSOA name", "YearMonth"] + \
    [f"lag1_{g}" for g in ["Anti-social", "Violence", "Theft", "Drugs", "Other"]]
lags = wide[lag_cols]
df = df.merge(lags, on=["LSOA name", "YearMonth"], how="left")

#  correct fill for lag columns
lag_cols_present = [col for col in df.columns if col.startswith("lag1_")]
for c in lag_cols_present:
    df[c] = df[c].fillna(0)

#  saved encoders
df["LSOA_encoded"] = lsoa_enc.transform(df["LSOA name"])
#  make y a pandas Series (fixes .loc error)
y = pd.Series(grp_enc.transform(df["Crime_group"]), index=df.index, name="y")

# Build features exactly as trained
try:
    raw_feats = pd.read_csv(
        FEATS_FILE, header=None).squeeze().dropna().astype(str).tolist()
    feat_list = []
    for ln in raw_feats:
        feat_list += [p.strip() for p in ln.split(",") if p.strip()]
    feat_list = [c for c in feat_list if c in df.columns]
except Exception:
    base_feats = ["Year", "Month_num", "Season",
                  "Latitude", "Longitude", "LSOA_encoded"]
    lag_feats = [f"lag1_{g}" for g in [
        "Anti-social", "Violence", "Theft", "Drugs", "Other"]]
    synth_feats = [c for c in [
        "last_month_total_crimes", "unemployment_rate", "median_income", "education_index",
        "population_density", "age_15_24_share", "urban_flag", "cctv_density", "street_lighting_score",
        "avg_temp_c", "avg_rain_mm", "events_index", "public_event_flag", "bank_holiday_flag",
        "policing_presence_index", "stop_search_rate_per_1k"
    ] if c in df.columns]
    feat_list = base_feats + lag_feats + synth_feats

X = df[feat_list].apply(pd.to_numeric, errors="coerce").fillna(0)

# Time-aware test split
test_mask = (df["Year"] == TEST_YEAR)
X_test = X.loc[test_mask].reset_index(drop=True)
y_test = y.loc[test_mask].astype(int).reset_index(drop=True)

print(f"Using {len(feat_list)} features.")
print(f"Test rows (Year={TEST_YEAR}): {len(X_test)}")

# Predict and  evaluate
proba = model.predict_proba(X_test)
y_pred = proba.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print(f"\n Accuracy: {acc:.2%}")
print(f" Macro F1: {f1m:.3f}")
if top_k_accuracy_score is not None:
    top3 = top_k_accuracy_score(y_test, proba, k=3)
    print(f" Top-3 Accuracy: {top3:.2%}")
else:
    top3 = np.nan

# Report and CM
report_txt = classification_report(y_test, y_pred, target_names=list(
    grp_enc.classes_), digits=3, zero_division=0)
print("\nClassification Report:")
print(report_txt)
cm = confusion_matrix(y_test, y_pred, labels=list(
    range(len(grp_enc.classes_))))
pd.DataFrame(cm, index=grp_enc.classes_, columns=grp_enc.classes_).to_csv(
    "results_confusion_matrix.csv")

#  Saving summaries
pd.DataFrame([{
    "Model": "Grouped XGB (Step3 retrain)",
    "Accuracy": acc,
    "Macro_F1": f1m,
    "Top3": top3,
    "N_test": len(X_test),
    "Features": len(feat_list)
}]).to_csv("results_summary.csv", index=False)

with open("results_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)

# CM PNG
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grp_enc.classes_).plot(
        include_values=False, cmap="Blues", xticks_rotation=90
    )
    plt.title(f"Confusion Matrix — Grouped ({TEST_YEAR})")
    plt.tight_layout()
    plt.savefig("results_confusion_matrix.png", dpi=150)
    plt.close()
    print(" Saved: results_confusion_matrix.png")
except Exception as e:
    print("(Plot skipped)", e)

print("\n Saved: results_summary.csv, results_classification_report.txt, results_confusion_matrix.csv")
