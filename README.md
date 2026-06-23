Predicting and Analysing Crime Types in Cambridgeshire Using Machine Learning
Overview

This project uses machine learning techniques to analyse historical crime data and predict crime categories across Cambridgeshire. The system combines spatial, temporal, and contextual features to identify crime patterns and improve crime-type prediction accuracy.

The project was developed using Python and incorporates data preprocessing, feature engineering, machine learning modelling, evaluation, and geospatial visualisation.

Objectives
Analyse crime trends in Cambridgeshire from 2017–2025.
Predict crime categories using machine learning models.
Compare Random Forest and XGBoost performance.
Investigate the impact of historical crime patterns using lag features.
Visualise crime hotspots through interactive maps.
Evaluate model effectiveness using multiple performance metrics.
Dataset

Source: UK Police Crime Data

Coverage:

Cambridgeshire
2017–2025

Features include:

Crime Type
Month and Year
Latitude and Longitude
LSOA Name
Seasonal Information
Historical Crime Counts (Lag Features)
Synthetic Contextual Variables
Data Processing

The following preprocessing steps were applied:

Data cleaning and validation
Missing value handling
Feature engineering
Label encoding
Crime category grouping
Temporal feature creation
Lag feature generation
Dataset merging and aggregation
Crime Category Grouping

Original crime types were grouped into five major categories:

Theft
Violence
Anti-social
Drugs
Other

This reduced class imbalance and improved model performance.

Machine Learning Models
Random Forest

Used as a baseline model for comparison.

XGBoost

Final model selected due to superior performance on structured crime data.

Parameters:

n_estimators = 600
max_depth = 8
learning_rate = 0.05
subsample = 0.9
colsample_bytree = 0.9
Results
Random Forest
Accuracy: ~20–26%
XGBoost (Grouped Crime Prediction)
Accuracy: 54.48%
Macro F1 Score: 0.237
Top-3 Accuracy: 89.55%

These results demonstrate a significant improvement over the baseline model.

Visualisations

The project includes:

Crime distribution charts
Class imbalance analysis
Confusion matrix
Feature importance plots
Crime hotspot heatmaps
Interactive LSOA crime maps

Generated using:

Matplotlib
Seaborn
Folium
Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Folium
Matplotlib
Seaborn
Joblib
Streamlit
Project Structure
├── datasets/
│   ├── crime_type_features.csv
│   └── crime_type_features_WITH_SYNTHETIC_CONTEXT.csv
│
├── models/
│   ├── crime_type_xgb_GROUPED_model.joblib
│   ├── crime_group_encoder.joblib
│   └── lsoa_encoder.joblib
│
├── maps/
│   └── crime_lsoa_areas_google_style_with_heatmap_2025.html
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── random_forest_model.py
│   ├── xgboost_grouped_model.py
│   └── map_generation.py
│
└── README.md
Running the Project

Install dependencies:

pip install pandas numpy scikit-learn xgboost folium matplotlib seaborn joblib streamlit imbalanced-learn

Train the model:

python xgboost_grouped_model.py

Generate maps:

python map_generation.py

Run Streamlit interface:

streamlit run step8.py
Future Improvements
Incorporate real socioeconomic datasets.
Experiment with deep learning approaches.
Add real-time crime prediction capabilities.
Improve minority class prediction performance.
Deploy as a web application.
Author

Dev Narayan

BSc Computer Science

University Final Year Project

2026

License

This project is for academic and research purposes only
