from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rainfall_7day_forecast import forecast_live_next_7_days, forecast_next_7_days


DATA_PATH = Path("up_daily_weather_dataset.csv")
MODEL_DIR = Path("models_7day")
EDA_DIR = Path("eda_outputs")
REQUIRED_MODEL_FILES = [
    "rainfall_7day_regressor.pkl",
    "rainfall_7day_classifier.pkl",
    "district_encoder.pkl",
    "feature_columns.pkl",
]


@st.cache_data
def load_districts(data_path: str) -> list[str]:
    df = pd.read_csv(data_path, usecols=["district"])
    return sorted(df["district"].dropna().unique().tolist())


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def models_are_available() -> bool:
    return all((MODEL_DIR / filename).exists() for filename in REQUIRED_MODEL_FILES)


def csv_or_none(path: Path) -> pd.DataFrame | None:
    return load_csv(str(path)) if path.exists() else None


def forecast_chart(forecast: pd.DataFrame) -> go.Figure:
    plot_data = forecast.copy()
    plot_data["forecast_date"] = pd.to_datetime(plot_data["forecast_date"])
    fig = go.Figure()
    fig.add_bar(
        x=plot_data["forecast_date"],
        y=plot_data["predicted_rainfall_mm"],
        name="Rainfall (mm)",
        marker_color="#3b82f6",
    )
    fig.add_scatter(
        x=plot_data["forecast_date"],
        y=plot_data["rain_probability"] * 100,
        name="Rain Probability (%)",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#f97316", width=3),
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(title="Rainfall (mm)", rangemode="tozero"),
        yaxis2=dict(title="Rain Probability (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def future_vulnerable_districts(candidates: list[str], threshold: float) -> pd.DataFrame:
    rows = []
    progress = st.progress(0)
    status = st.empty()
    for idx, district in enumerate(candidates, start=1):
        status.write(f"Forecasting {district}...")
        try:
            forecast = forecast_live_next_7_days(
                output_dir=MODEL_DIR,
                place=district,
                rain_probability_threshold=threshold,
            )
            rows.append(
                {
                    "district": forecast["matched_model_district"].iloc[0],
                    "total_predicted_rainfall_mm": forecast["predicted_rainfall_mm"].sum(),
                    "rainy_days_next_7": int((forecast["rain_prediction"] == "Rain").sum()),
                    "avg_rain_probability_pct": forecast["rain_probability"].mean() * 100,
                    "max_daily_rainfall_mm": forecast["predicted_rainfall_mm"].max(),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "district": district,
                    "total_predicted_rainfall_mm": None,
                    "rainy_days_next_7": None,
                    "avg_rain_probability_pct": None,
                    "max_daily_rainfall_mm": None,
                    "error": str(exc),
                }
            )
        progress.progress(idx / len(candidates))
    status.empty()
    progress.empty()
    result = pd.DataFrame(rows)
    if "total_predicted_rainfall_mm" in result:
        result = result.sort_values("total_predicted_rainfall_mm", ascending=False, na_position="last")
    return result


st.set_page_config(page_title="7-Day Rainfall Intelligence", layout="wide")
st.title("7-Day Rainfall Intelligence")

if not DATA_PATH.exists():
    st.error("Dataset not found: up_daily_weather_dataset.csv")
    st.stop()

districts = load_districts(str(DATA_PATH))
forecast_tab, risk_tab, model_tab = st.tabs(["Forecast", "Rainfall Risk", "Model Performance"])

with forecast_tab:
    left, right = st.columns([0.28, 0.72], gap="large")

    with left:
        use_live_data = st.toggle("Use live weather data", value=True)
        selected_district = st.selectbox("District", districts, index=districts.index("Agra") if "Agra" in districts else 0)
        place = st.text_input("Place or district name", value=selected_district)
        threshold = st.slider("Rain probability cutoff", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        run_forecast = st.button("Forecast", type="primary", use_container_width=True)

    with right:
        if not models_are_available():
            st.warning("Train the model first to enable prediction.")
            st.code("python rainfall_7day_forecast.py train --include-lstm", language="powershell")
        elif run_forecast:
            try:
                if use_live_data:
                    forecast = forecast_live_next_7_days(
                        output_dir=MODEL_DIR,
                        place=place.strip(),
                        rain_probability_threshold=threshold,
                    )
                    matched_district = forecast["matched_model_district"].iloc[0]
                else:
                    forecast = forecast_next_7_days(
                        data_path=DATA_PATH,
                        output_dir=MODEL_DIR,
                        district=place.strip(),
                        rain_probability_threshold=threshold,
                    )
                    matched_district = place.strip()
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            total_rain = forecast["predicted_rainfall_mm"].sum()
            rainy_days = int((forecast["rain_prediction"] == "Rain").sum())
            avg_probability = forecast["rain_probability"].mean() * 100

            metric_cols = st.columns(4)
            metric_cols[0].metric("Model District", matched_district)
            metric_cols[1].metric("7-Day Rainfall", f"{total_rain:.2f} mm")
            metric_cols[2].metric("Rainy Days", f"{rainy_days} / 7")
            metric_cols[3].metric("Avg Rain Chance", f"{avg_probability:.1f}%")

            st.plotly_chart(forecast_chart(forecast), use_container_width=True)

            display = forecast.copy()
            display["predicted_rainfall_mm"] = display["predicted_rainfall_mm"].round(2)
            display["rain_probability"] = (display["rain_probability"] * 100).round(1)
            display_columns = [
                "forecast_date",
                "day_ahead",
                "predicted_rainfall_mm",
                "rain_prediction",
                "rain_probability",
            ]
            weather_columns = [
                "temperature_mean_c",
                "temperature_max_c",
                "temperature_min_c",
                "humidity_pct",
                "wind_speed_max_ms",
                "wind_gusts_max_ms",
                "surface_pressure_pa",
                "cloud_cover_pct",
                "open_meteo_precipitation_probability_pct",
            ]
            display_columns.extend([column for column in weather_columns if column in display.columns])
            st.dataframe(display[display_columns], hide_index=True, use_container_width=True)
        else:
            st.info("Choose a place and click Forecast.")

with risk_tab:
    district_risk = csv_or_none(EDA_DIR / "eda_district_risk.csv")
    recent_rainfall = csv_or_none(EDA_DIR / "eda_district_rainfall_last_14_years.csv")
    monthly_risk = csv_or_none(EDA_DIR / "eda_monthly_risk.csv")
    district_month = csv_or_none(EDA_DIR / "eda_district_month_risk.csv")

    if district_risk is None or recent_rainfall is None:
        st.warning("Run EDA first to populate this dashboard.")
        st.code("python rainfall_7day_forecast.py eda", language="powershell")
    else:
        top_recent = recent_rainfall.head(15)
        top_vulnerable = district_risk.head(15)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Highest Total Rainfall: Last 14 Years")
            fig = px.bar(
                top_recent.sort_values("total_rainfall_last_14_years_mm"),
                x="total_rainfall_last_14_years_mm",
                y="district",
                orientation="h",
                labels={"total_rainfall_last_14_years_mm": "Total Rainfall (mm)", "district": "District"},
                color="heavy_rain_days",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Most Vulnerable Districts")
            fig = px.bar(
                top_vulnerable.sort_values("district_risk_score_0_100"),
                x="district_risk_score_0_100",
                y="district",
                orientation="h",
                labels={"district_risk_score_0_100": "Risk Score", "district": "District"},
                color="rainy_day_pct",
                color_continuous_scale="Greens",
            )
            st.plotly_chart(fig, use_container_width=True)

        if monthly_risk is not None:
            st.subheader("High-Risk Months")
            month_fig = px.line(
                monthly_risk,
                x="month_name",
                y=["mean_daily_rainfall_mm", "rainy_day_pct"],
                markers=True,
                labels={"value": "Value", "month_name": "Month", "variable": "Metric"},
            )
            st.plotly_chart(month_fig, use_container_width=True)

        if district_month is not None:
            st.subheader("District-Month Rainfall Heatmap")
            selected = st.multiselect(
                "Districts",
                district_risk["district"].head(30).tolist(),
                default=district_risk["district"].head(10).tolist(),
            )
            heatmap_data = district_month[district_month["district"].isin(selected)]
            heatmap = heatmap_data.pivot(index="district", columns="month_name", values="mean_daily_rainfall_mm")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            heatmap = heatmap[[month for month in month_order if month in heatmap.columns]]
            fig = px.imshow(
                heatmap,
                aspect="auto",
                color_continuous_scale="YlGnBu",
                labels=dict(color="Mean Rainfall (mm)", x="Month", y="District"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Future Vulnerable Districts: Next 7 Days")
        st.caption("This uses live Open-Meteo forecasts for the historically vulnerable districts you choose below.")
        count = st.slider("Number of districts to scan", min_value=3, max_value=20, value=8)
        threshold = st.slider("Rain probability cutoff for future scan", 0.1, 0.9, 0.5, 0.05)
        candidates = district_risk["district"].head(count).tolist()
        if st.button("Generate Future Vulnerability Ranking", use_container_width=True):
            if not models_are_available():
                st.error("Train the model before generating future vulnerability.")
            else:
                future_risk = future_vulnerable_districts(candidates, threshold)
                st.dataframe(future_risk.round(2), hide_index=True, use_container_width=True)
                clean = future_risk.dropna(subset=["total_predicted_rainfall_mm"])
                if not clean.empty:
                    fig = px.bar(
                        clean.sort_values("total_predicted_rainfall_mm"),
                        x="total_predicted_rainfall_mm",
                        y="district",
                        orientation="h",
                        color="avg_rain_probability_pct",
                        color_continuous_scale="Blues",
                        labels={
                            "total_predicted_rainfall_mm": "Next 7 Days Rainfall (mm)",
                            "district": "District",
                            "avg_rain_probability_pct": "Avg Rain Probability (%)",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)

with model_tab:
    comparison = csv_or_none(MODEL_DIR / "model_comparison.csv")
    overall = csv_or_none(MODEL_DIR / "7day_overall_train_validation_test_metrics.csv")
    regression = csv_or_none(MODEL_DIR / "7day_regression_metrics_by_horizon.csv")
    classification = csv_or_none(MODEL_DIR / "7day_classification_metrics_by_horizon.csv")
    xgb_history = csv_or_none(MODEL_DIR / "xgboost_training_history.csv")
    lstm_history = csv_or_none(MODEL_DIR / "lstm_training_history.csv")

    if comparison is None and overall is None:
        st.warning("Train the model first to populate performance charts.")
        st.code("python rainfall_7day_forecast.py train --include-lstm", language="powershell")
    else:
        if comparison is not None:
            st.subheader("Model Comparison")
            st.dataframe(comparison.round(4), hide_index=True, use_container_width=True)
            fig = px.bar(comparison, x="model", y=["mae_mm", "rmse_mm"], barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        if overall is not None:
            st.subheader("Train / Validation / Test Summary")
            st.dataframe(overall.round(4), hide_index=True, use_container_width=True)

        if regression is not None and classification is not None:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                fig = px.line(
                    regression,
                    x="horizon_day",
                    y="mae_mm",
                    color="split",
                    markers=True,
                    labels={"horizon_day": "Day Ahead", "mae_mm": "MAE (mm)"},
                    title="Regression Error by Forecast Day",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(
                    classification,
                    x="horizon_day",
                    y="accuracy",
                    color="split",
                    markers=True,
                    labels={"horizon_day": "Day Ahead", "accuracy": "Accuracy"},
                    title="Rain / No-Rain Accuracy by Forecast Day",
                )
                st.plotly_chart(fig, use_container_width=True)

        if xgb_history is not None:
            st.subheader("XGBoost Training Curves")
            reg_loss = xgb_history[(xgb_history["model"] == "xgboost_regressor") & (xgb_history["metric"] == "rmse")]
            cls_loss = xgb_history[(xgb_history["model"] == "xgboost_classifier") & (xgb_history["metric"] == "logloss")]
            cls_error = xgb_history[(xgb_history["model"] == "xgboost_classifier") & (xgb_history["metric"] == "error")].copy()
            if not reg_loss.empty:
                fig = px.line(reg_loss, x="iteration", y="value", color="split", title="Regression Loss")
                st.plotly_chart(fig, use_container_width=True)
            if not cls_loss.empty:
                fig = px.line(cls_loss, x="iteration", y="value", color="split", title="Classification Log Loss")
                st.plotly_chart(fig, use_container_width=True)
            if not cls_error.empty:
                cls_error["accuracy"] = 1 - cls_error["value"]
                fig = px.line(
                    cls_error,
                    x="iteration",
                    y="accuracy",
                    color="split",
                    title="Classification Accuracy",
                    labels={"accuracy": "Accuracy", "iteration": "Boosting Round"},
                )
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

        if lstm_history is not None:
            st.subheader("LSTM Training Curves")
            melted = lstm_history.melt(id_vars="epoch", var_name="metric", value_name="value")
            selected_metrics = [metric for metric in melted["metric"].unique() if "loss" in metric or "accuracy" in metric]
            metric_choice = st.multiselect("LSTM metrics", selected_metrics, default=selected_metrics[:4])
            fig = px.line(melted[melted["metric"].isin(metric_choice)], x="epoch", y="value", color="metric")
            st.plotly_chart(fig, use_container_width=True)
