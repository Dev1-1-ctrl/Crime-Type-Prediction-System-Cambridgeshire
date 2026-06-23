# 🚔 Predicting & Analysing Crime Types in Cambridgeshire

> A machine learning system for predicting crime categories using historical UK Police data, spatial features, and temporal lag patterns.
![Python](https://img.shields.io/badge/Python-3.10+-blue)

![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

![Status](https://img.shields.io/badge/Status-Completed-success)

![License](https://img.shields.io/badge/License-Academic-orange)
---

## 📌 Overview

This project analyses **645,000+ historical crime records** from Cambridgeshire (2017–2025) and applies machine learning to predict crime categories using temporal, spatial, and contextual features.

The system combines:

- Historical UK Police crime records
- Geographical (LSOA-level) information
- Feature engineering & lag-based crime history
- XGBoost classification
- Interactive crime hotspot maps

---

## 🗺️ Crime Analysis Map

<!-- Replace with your actual screenshot -->
![Crime Heatmap](images/crime_heatmap.png)

---

## 🎯 Objectives

- Analyse crime trends across Cambridgeshire (2017–2025)
- Predict crime categories using machine learning
- Compare Random Forest vs XGBoost performance
- Investigate the impact of historical lag features
- Visualise crime hotspots using interactive maps

---

## 📊 Dataset

**Source:** [UK Police Crime Data](https://data.police.uk/)

| Property | Detail |
|----------|--------|
| Location | Cambridgeshire, UK |
| Period | 2017–2025 |
| Records | 645,000+ incidents |

**Key Features:**

| Feature | Description |
|---------|-------------|
| `Crime Type` | Recorded offence category |
| `Month / Year` | Crime occurrence date |
| `Latitude / Longitude` | Crime location coordinates |
| `LSOA Name` | Geographic area identifier |
| `Season` | Derived seasonal feature |
| `Lag Features` | Previous month's crime activity |
| `Synthetic Context` | Socioeconomic & environmental indicators |

---

## ⚙️ Pipeline

```
Raw Crime Data
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Crime Grouping (14 → 5 classes)
    │
    ▼
Lag Feature Creation
    │
    ▼
Model Training (XGBoost)
    │
    ▼
Evaluation
    │
    ▼
Interactive Map Generation
```

---

## 🔍 Crime Grouping

The original 14 crime categories were consolidated into 5 broader groups to reduce class imbalance:

| Group | Original Crime Types |
|-------|---------------------|
| **Theft** | Burglary, Robbery, Vehicle Crime, Shoplifting |
| **Violence** | Violence, Weapons, Arson |
| **Anti-social** | Anti-social Behaviour |
| **Drugs** | Drugs |
| **Other** | Other Crime, Public Order |

---

## 🤖 Models

### Random Forest (Baseline)
- Accuracy: ~20–26%

### XGBoost (Final Model)

```python
XGBClassifier(
    n_estimators      = 600,
    max_depth         = 8,
    learning_rate     = 0.05,
    subsample         = 0.9,
    colsample_bytree  = 0.9
)
```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | **54.48%** |
| Macro F1 Score | **0.237** |
| Top-3 Accuracy | **89.55%** |

**Key Findings:**

- ✅ Significant improvement over Random Forest baseline
- ✅ Strong performance for dominant crime groups (Theft, Violence)
- ✅ Temporal and spatial features proved highly informative
- ✅ Historical lag features meaningfully improved predictions

---


## 🌍 Interactive Crime Maps

The project generates interactive HTML maps including:

- Crime hotspot heatmaps
- LSOA-level crime density analysis
- Geographic clustering
- Web-based interactive viewer

Open `maps/crime_lsoa_areas_google_style_with_heatmap_2025.html` in a browser to explore.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, Imbalanced-learn |
| Visualisation | Matplotlib, Seaborn, Folium |
| Model Persistence | Joblib |

---

## 📂 Project Structure

```
Crime-Prediction-Project/
│
├── datasets/
│   ├── crime_type_features.csv
│   └── crime_type_features_WITH_SYNTHETIC_CONTEXT.csv
│
├── models/
│   ├── crime_type_xgb_GROUPED_model.joblib
│   ├── lsoa_encoder.joblib
│   └── crime_group_encoder.joblib
│
├── maps/
│   └── crime_lsoa_areas_google_style_with_heatmap_2025.html
│
├── images/
│   ├── confusion_matrix.png
│   ├── crime_heatmap.png
│   ├── class_distribution_compare_smote.png
│   └── lsoa_map.png
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── random_forest_model.py
│   ├── xgboost_grouped_model.py
│   └── map_generation.py
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/crime-prediction-cambridgeshire.git
cd crime-prediction-cambridgeshire
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn folium imbalanced-learn joblib
```

### 3. Train the model

```bash
python scripts/xgboost_grouped_model.py
```

### 4. Generate crime maps

```bash
python scripts/map_generation.py
```

---

## 👨‍💻 Author

**Dev Narayan**
BSc (Hons) Computer Science — Final Year Project
University of Bedfordshire

---

## 📜 Academic Notice

This project was developed as part of a final-year undergraduate dissertation in Computer Science at the University of Bedfordshire. It demonstrates machine learning, data analytics, and geospatial visualisation applied to crime prediction using historical UK Police data.

**Intended for academic, research, and educational purposes only.**
