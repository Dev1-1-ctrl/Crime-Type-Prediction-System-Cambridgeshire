import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load train/test splits
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# Load encoders for decoding crime types
crime_enc = joblib.load("crime_type_encoder.joblib")

# Train XGBoost
model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=1  # imbalance handled internally
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print(f" Accuracy: {acc:.2%}")
print(f" Macro F1: {f1_macro:.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=crime_enc.classes_,
    digits=3
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=crime_enc.classes_)
plt.figure(figsize=(12, 10))
disp.plot(include_values=False, cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix — Crime Type (Test: 2025)")
plt.tight_layout()
plt.show()

# Feature Importance
importances = model.feature_importances_
feat_names = X_train.columns
fi = pd.DataFrame({"feature": feat_names, "importance": importances}
                  ).sort_values("importance", ascending=False)

print("\nTop features:\n", fi.head(10))

plt.figure(figsize=(8, 5))
plt.barh(fi["feature"], fi["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance — XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Saving the model
joblib.dump(model, "crime_type_xgb_model.joblib")
print(" Model saved → crime_type_xgb_model.joblib")
