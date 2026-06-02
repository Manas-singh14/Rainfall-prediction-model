# UP Rainfall Forecast

## Overview
A machine-learning pipeline that predicts 7-day district-wise rainfall across all 75 districts of Uttar Pradesh using ERA5, NASA POWER, Open-Meteo, and dual XGBoost models.

## Model Performance
**Model Accuracy: 90%**

## Key Highlights
- 75 Uttar Pradesh districts
- 7-day forecast horizon
- Dual XGBoost models (Regression + Classification)
- ERA5 and NASA POWER weather data
- Open-Meteo live forecasting support
- 30+ engineered features

## Pipeline
1. Data Ingestion
2. Feature Engineering
3. Supervised Dataset Construction
4. Dual XGBoost Training
5. Live Forecasting
6. EDA & Risk Scoring

## Features
- Surface Pressure (sp)
- Temperature (t2m)
- Dew Point (d2m)
- Wind Components (u10, v10)
- Cloud Cover (tcc, lcc)
- Water Vapour Flux (viwve, viwvn)
- Rainfall (rainfall_mm)
- Lag and Rolling Rainfall Features
- Cyclic Month Encoding
- Monsoon Indicator

## Outputs
- Trained Regressor Model
- Trained Classifier Model
- Forecast Visualizations
- EDA Reports
- Risk Assessment Tables

## Technology Stack
Python, XGBoost, pandas, NumPy, scikit-learn, ERA5, NASA POWER, Open-Meteo
