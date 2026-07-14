# UP Rainfall Forecast

## Overview
A machine-learning pipeline that predicts 7-day district-wise rainfall across all 75 districts of Uttar Pradesh using ERA5, NASA POWER, Open-Meteo, and dual XGBoost models.

## Model Performance
**Model Accuracy: 90%**
## Results
<img width="1055" height="603" alt="image" src="https://github.com/user-attachments/assets/71060c90-e916-4cb4-a5bf-4aff555cf5b5" />
<img width="993" height="630" alt="image" src="https://github.com/user-attachments/assets/ed4461b9-55b5-491e-81a0-8f3f4cda65a1" />

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
