 🚔 Predicting and Analysing Crime Types in Cambridgeshire Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Academic-orange)

## 📌 Project Overview

Crime prediction is an important area of data analytics that helps identify patterns, understand crime trends, and support evidence-based decision making.

This project analyses historical crime data from Cambridgeshire (2017–2025) and applies machine learning techniques to predict crime categories using temporal, spatial, and contextual features.

The system combines:

- Historical crime records
- Geographical information
- Feature engineering
- Lag-based crime history
- Machine learning models
- Crime hotspot visualisation

---

## 🗺️ Crime Analysis Map

Insert your crime map screenshot here.

![Crime Heatmap](images/crime_heatmap.png)

---

## 🎯 Objectives

- Analyse crime trends across Cambridgeshire
- Predict crime categories using machine learning
- Compare Random Forest and XGBoost performance
- Investigate the impact of historical crime patterns
- Visualise crime hotspots using interactive maps
- Evaluate prediction performance using multiple metrics

---

## 📊 Dataset

### Source

UK Police Crime Data

### Coverage

- Location: Cambridgeshire, UK
- Period: 2017–2025
- Records: 645,000+ crime incidents

### Main Features

| Feature | Description |
|----------|-------------|
| Crime Type | Recorded offence category |
| Month | Crime occurrence month |
| Year | Crime occurrence year |
| Latitude | Crime location latitude |
| Longitude | Crime location longitude |
| LSOA Name | Geographic area identifier |
| Season | Derived seasonal feature |
| Lag Features | Previous month's crime activity |
| Synthetic Context | Socioeconomic and environmental indicators |

---

## ⚙️ Data Processing Pipeline

```text
Raw Crime Data
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Crime Grouping
       │
       ▼
Lag Feature Creation
       │
       ▼
Model Training
       │
       ▼
Evaluation
       │
       ▼
Visualisation
🔍 Crime Grouping

The original 14 crime categories were grouped into 5 broader classes:

Original Crime Types	Group
Burglary, Robbery, Vehicle Crime, Shoplifting	Theft
Violence, Weapons, Arson	Violence
Anti-social Behaviour	Anti-social
Drugs	Drugs
Other Crime, Public Order	Other

This reduced class imbalance and improved prediction performance.

🤖 Machine Learning Models
Random Forest

Used as the baseline model.

Accuracy: ~20–26%

XGBoost (Final Model)

Selected due to superior performance on structured crime data.

Parameters:

n_estimators = 600
max_depth = 8
learning_rate = 0.05
subsample = 0.9
colsample_bytree = 0.9
📈 Results
Final XGBoost Model Performance
Metric	Score
Accuracy	54.48%
Macro F1 Score	0.237
Top-3 Accuracy	89.55%
Key Findings

✅ Significant improvement over Random Forest

✅ Strong performance for major crime groups

✅ Effective use of temporal and spatial features

✅ Historical lag features improved predictions

📊 Model Evaluation
Confusion Matrix

Insert your confusion matrix screenshot.

Class Distribution

Insert your SMOTE comparison graph.

🌍 Interactive Crime Visualisation

The project includes:

Crime hotspot heatmaps
LSOA-level crime analysis
Geographic crime clustering
Interactive web-based maps
Example

🛠 Technologies Used
Programming
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
Imbalanced-learn
Visualisation
Matplotlib
Seaborn
Folium
Model Persistence
Joblib
📂 Project Structure
Crime-Prediction-Project
│
├── datasets
│   ├── crime_type_features.csv
│   └── crime_type_features_WITH_SYNTHETIC_CONTEXT.csv
│
├── models
│   ├── crime_type_xgb_GROUPED_model.joblib
│   ├── lsoa_encoder.joblib
│   └── crime_group_encoder.joblib
│
├── maps
│   └── crime_lsoa_areas_google_style_with_heatmap_2025.html
│
├── images
│   ├── confusion_matrix.png
│   ├── crime_heatmap.png
│   ├── class_distribution_compare_smote.png
│   └── lsoa_map.png
│
├── scripts
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── random_forest_model.py
│   ├── xgboost_grouped_model.py
│   └── map_generation.py
│
└── README.md
🚀 Running the Project
Install Dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn folium imbalanced-learn joblib
Train Model
python xgboost_grouped_model.py
Generate Crime Maps
python map_generation.py
🔮 Future Work
Incorporate real socioeconomic datasets
Improve minority crime prediction
Explore deep learning approaches
Deploy as a web application
Integrate real-time crime data
👨‍💻 Author

Dev Narayan

Final Year Project

BSc Computer Science

University of Bedfordshire

📜 License

This project was developed for academic and research purposes.


# What you should add

Create an `images` folder and add:

1. `crime_heatmap.png` ← screenshot of your Folium heatmap
2. `confusion_matrix.png`
3. `class_distribution_compare_smote.png`
4. `lsoa_map.png`

The screenshots make a huge difference. A repository with visuals gets far more attention than one with only text.

Also add your actual results near the top (54.48% accuracy, 89.55% top-3 accuracy) because that's what people look 
