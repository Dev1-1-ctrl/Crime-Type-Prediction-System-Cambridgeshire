import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset which cleaned
df = pd.read_csv("crime_type_features_WITH_SYNTHETIC_CONTEXT.csv")

# Encode categorical features
lsoa_encoder = LabelEncoder()
df["LSOA_encoded"] = lsoa_encoder.fit_transform(df["LSOA name"])

crime_encoder = LabelEncoder()
df["Crime_encoded"] = crime_encoder.fit_transform(df["Crime type"])

# Feature selection
features = [
    "Year", "Month_num", "Season",
    "Latitude", "Longitude", "LSOA_encoded",
    "last_month_total_crimes", "unemployment_rate", "median_income",
    "education_index", "population_density", "age_15_24_share",
    "urban_flag", "cctv_density", "street_lighting_score",
    "avg_temp_c", "avg_rain_mm", "events_index", "public_event_flag",
    "bank_holiday_flag", "policing_presence_index", "stop_search_rate_per_1k"
]
X = df[features]
y = df["Crime_encoded"]

# Time-aware split
X_train = X[df["Year"] <= 2024]
y_train = y[df["Year"] <= 2024]
X_test = X[df["Year"] == 2025]
y_test = y[df["Year"] == 2025]

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print(f" Accuracy: {acc:.2%}")
print(f" Macro F1: {f1m:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=crime_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)

labels = ["Anti-social", "Drugs", "Other", "Theft", "Violence"]

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,          # no numbers inside the cells
    cbar=True,
    square=False,
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="white"
)

plt.title("Confusion Matrix — Grouped (2025)", fontsize=16)
plt.xlabel("Predicted label", fontsize=12)
plt.ylabel("True label", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("rf_confusion_matrix_grouped_2025.png", dpi=300)
plt.show()
