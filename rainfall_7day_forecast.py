from __future__ import annotations

import argparse
import calendar
import difflib
import json
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


RAW_WEATHER_COLUMNS = [
    "sp",
    "tcc",
    "u10",
    "v10",
    "t2m",
    "d2m",
    "lcc",
    "viwve",
    "viwvn",
]

LAGS = [1, 2, 3, 7, 14]
ROLL_WINDOWS = [3, 7, 14, 30]
HORIZONS = range(1, 8)
RAIN_THRESHOLD_MM = 0.1
HEAVY_RAIN_MM = 64.5
VERY_HEAVY_RAIN_MM = 115.6
EXTREME_RAIN_MM = 204.5
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "total_column_integrated_water_vapour",
]
OPEN_METEO_DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]


def load_weather(path: Path, rain_cap_quantile: float = 0.99) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"district", "date", "rainfall_mm", *RAW_WEATHER_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    if df["t2m"].mean() > 100:
        df["t2m"] = df["t2m"] - 273.15
        df["d2m"] = df["d2m"] - 273.15

    if rain_cap_quantile:
        cap = df["rainfall_mm"].quantile(rain_cap_quantile)
        df["rainfall_mm"] = df["rainfall_mm"].clip(upper=cap)

    return df


def add_as_of_features(df: pd.DataFrame, encoder: LabelEncoder | None = None) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear

    df["dewpoint_depression"] = df["t2m"] - df["d2m"]
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["moisture_flux"] = np.sqrt(df["viwve"] ** 2 + df["viwvn"] ** 2)
    df["is_monsoon"] = df["month"].between(6, 9).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366)

    grouped_rain = df.groupby("district")["rainfall_mm"]
    for lag in LAGS:
        df[f"rain_lag_{lag}"] = grouped_rain.shift(lag)

    for window in ROLL_WINDOWS:
        shifted = grouped_rain.shift(1)
        df[f"rain_roll{window}_mean"] = shifted.groupby(df["district"]).rolling(window).mean().reset_index(level=0, drop=True)
        df[f"rain_roll{window}_sum"] = shifted.groupby(df["district"]).rolling(window).sum().reset_index(level=0, drop=True)

    if encoder is None:
        encoder = LabelEncoder()
        df["district_enc"] = encoder.fit_transform(df["district"])
    else:
        unknown = sorted(set(df["district"]) - set(encoder.classes_))
        if unknown:
            raise ValueError(f"Unknown districts for saved encoder: {unknown[:5]}")
        df["district_enc"] = encoder.transform(df["district"])

    return df, encoder


def feature_columns() -> list[str]:
    lag_cols = [f"rain_lag_{lag}" for lag in LAGS]
    roll_cols = []
    for window in ROLL_WINDOWS:
        roll_cols.extend([f"rain_roll{window}_mean", f"rain_roll{window}_sum"])

    return [
        *RAW_WEATHER_COLUMNS,
        "dewpoint_depression",
        "wind_speed",
        "moisture_flux",
        *lag_cols,
        *roll_cols,
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_monsoon",
        "district_enc",
        "horizon",
        "target_month_sin",
        "target_month_cos",
        "target_dayofyear_sin",
        "target_dayofyear_cos",
    ]


def make_supervised_7day(df: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    rows = []
    base = df.copy()
    weather_cols = [
        *RAW_WEATHER_COLUMNS,
        "dewpoint_depression",
        "wind_speed",
        "moisture_flux",
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_monsoon",
    ]

    for horizon in horizons:
        part = base.copy()
        target_date = part["date"] + pd.to_timedelta(horizon, unit="D")
        target_dayofyear = target_date.dt.dayofyear
        target_month = target_date.dt.month

        part["horizon"] = horizon
        part["target_date"] = target_date
        part["target_month_sin"] = np.sin(2 * np.pi * target_month / 12)
        part["target_month_cos"] = np.cos(2 * np.pi * target_month / 12)
        part["target_dayofyear_sin"] = np.sin(2 * np.pi * target_dayofyear / 366)
        part["target_dayofyear_cos"] = np.cos(2 * np.pi * target_dayofyear / 366)
        part["target_rainfall_mm"] = part.groupby("district")["rainfall_mm"].shift(-horizon)
        part["target_rain_occurred"] = (part["target_rainfall_mm"] > RAIN_THRESHOLD_MM).astype(int)

        # Use target-date weather as a stand-in for live forecast weather.
        # Rainfall lag/rolling features stay as-of the prediction date.
        for column in weather_cols:
            part[column] = part.groupby("district")[column].shift(-horizon)

        rows.append(part)

    supervised = pd.concat(rows, ignore_index=True)
    needed = feature_columns() + ["target_rainfall_mm"]
    return supervised.dropna(subset=needed).reset_index(drop=True)


def make_models(random_state: int) -> tuple[XGBRegressor, XGBClassifier]:
    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=random_state,
        eval_metric="rmse",
        verbosity=0,
    )
    classifier = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=random_state,
        eval_metric=["logloss", "error"],
        verbosity=0,
    )
    return regressor, classifier


def evaluate_regression_by_horizon(data: pd.DataFrame, predictions: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        mask = data["horizon"] == horizon
        actual = data.loc[mask, "target_rainfall_mm"]
        pred = predictions[mask]
        rows.append(
            {
                "split": split_name,
                "horizon_day": horizon,
                "mae_mm": mean_absolute_error(actual, pred),
                "rmse_mm": np.sqrt(mean_squared_error(actual, pred)),
                "r2": r2_score(actual, pred),
                "rows": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_classification_by_horizon(data: pd.DataFrame, predictions: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        mask = data["horizon"] == horizon
        actual = data.loc[mask, "target_rain_occurred"]
        pred = predictions[mask]
        rows.append(
            {
                "split": split_name,
                "horizon_day": horizon,
                "accuracy": accuracy_score(actual, pred),
                "rows": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def save_evaluation_plots(
    regression_metrics: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    test_data: pd.DataFrame,
    test_predictions: np.ndarray,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot_mae = regression_metrics.pivot(index="horizon_day", columns="split", values="mae_mm")
    pivot_rmse = regression_metrics.pivot(index="horizon_day", columns="split", values="rmse_mm")
    pivot_r2 = regression_metrics.pivot(index="horizon_day", columns="split", values="r2")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    pivot_mae.plot(ax=axes[0], marker="o")
    axes[0].set_title("MAE by Forecast Day")
    axes[0].set_xlabel("Day Ahead")
    axes[0].set_ylabel("MAE (mm)")

    pivot_rmse.plot(ax=axes[1], marker="o")
    axes[1].set_title("RMSE by Forecast Day")
    axes[1].set_xlabel("Day Ahead")
    axes[1].set_ylabel("RMSE (mm)")

    pivot_r2.plot(ax=axes[2], marker="o")
    axes[2].set_title("R2 by Forecast Day")
    axes[2].set_xlabel("Day Ahead")
    axes[2].set_ylabel("R2")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title="Split")

    fig.tight_layout()
    fig.savefig(output_dir / "regression_train_test_metrics.png", dpi=160)
    plt.close(fig)

    pivot_acc = classification_metrics.pivot(index="horizon_day", columns="split", values="accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_acc.plot(ax=ax, marker="o")
    ax.set_title("Rain / No Rain Accuracy by Forecast Day")
    ax.set_xlabel("Day Ahead")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(output_dir / "classification_train_test_accuracy.png", dpi=160)
    plt.close(fig)

    sample_size = min(20000, len(test_data))
    plot_data = test_data[["target_rainfall_mm", "horizon"]].copy()
    plot_data["predicted_rainfall_mm"] = test_predictions
    if sample_size < len(plot_data):
        plot_data = plot_data.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        plot_data["target_rainfall_mm"],
        plot_data["predicted_rainfall_mm"],
        c=plot_data["horizon"],
        cmap="viridis",
        alpha=0.25,
        s=8,
    )
    max_value = max(plot_data["target_rainfall_mm"].max(), plot_data["predicted_rainfall_mm"].max())
    ax.plot([0, max_value], [0, max_value], color="red", linestyle="--", linewidth=1)
    ax.set_title("Actual vs Predicted Rainfall on Test Set")
    ax.set_xlabel("Actual Rainfall (mm)")
    ax.set_ylabel("Predicted Rainfall (mm)")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Day Ahead")
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted_test.png", dpi=160)
    plt.close(fig)


def add_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    eda = df.copy()
    eda["year"] = eda["date"].dt.year
    eda["month"] = eda["date"].dt.month
    eda["month_name"] = eda["month"].map(lambda month: calendar.month_abbr[month])
    eda["rainy_day"] = eda["rainfall_mm"] > RAIN_THRESHOLD_MM
    eda["heavy_rain_day"] = eda["rainfall_mm"] >= HEAVY_RAIN_MM
    eda["very_heavy_rain_day"] = eda["rainfall_mm"] >= VERY_HEAVY_RAIN_MM
    eda["extreme_rain_day"] = eda["rainfall_mm"] >= EXTREME_RAIN_MM
    return eda


def percentile_score(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=series.index)
    return series.rank(pct=True) * 100


def build_eda_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eda = add_eda_columns(df)
    max_year = int(eda["year"].max())
    recent_start_year = max_year - 13
    recent_eda = eda[eda["year"].between(recent_start_year, max_year)].copy()

    district_year = (
        eda.groupby(["district", "year"])
        .agg(
            annual_rainfall_mm=("rainfall_mm", "sum"),
            rainy_days=("rainy_day", "sum"),
            heavy_rain_days=("heavy_rain_day", "sum"),
            max_daily_rainfall_mm=("rainfall_mm", "max"),
        )
        .reset_index()
    )

    yearly_risk = (
        eda.groupby("year")
        .agg(
            mean_daily_rainfall_mm=("rainfall_mm", "mean"),
            total_rainfall_mm=("rainfall_mm", "sum"),
            rainy_day_pct=("rainy_day", "mean"),
            heavy_rain_days=("heavy_rain_day", "sum"),
            very_heavy_rain_days=("very_heavy_rain_day", "sum"),
            extreme_rain_days=("extreme_rain_day", "sum"),
            max_daily_rainfall_mm=("rainfall_mm", "max"),
        )
        .reset_index()
    )
    annual_by_district = district_year.groupby("year")["annual_rainfall_mm"]
    yearly_risk["avg_district_annual_rainfall_mm"] = yearly_risk["year"].map(annual_by_district.mean())
    yearly_risk["max_district_annual_rainfall_mm"] = yearly_risk["year"].map(annual_by_district.max())
    yearly_risk["rainy_day_pct"] = yearly_risk["rainy_day_pct"] * 100
    yearly_risk["year_risk_score_0_100"] = (
        percentile_score(yearly_risk["avg_district_annual_rainfall_mm"])
        + percentile_score(yearly_risk["heavy_rain_days"])
        + percentile_score(yearly_risk["max_daily_rainfall_mm"])
    ) / 3
    yearly_risk = yearly_risk.sort_values("year")

    monthly_risk = (
        eda.groupby(["month", "month_name"])
        .agg(
            mean_daily_rainfall_mm=("rainfall_mm", "mean"),
            median_daily_rainfall_mm=("rainfall_mm", "median"),
            total_rainfall_mm=("rainfall_mm", "sum"),
            rainy_day_pct=("rainy_day", "mean"),
            heavy_rain_days=("heavy_rain_day", "sum"),
            very_heavy_rain_days=("very_heavy_rain_day", "sum"),
            extreme_rain_days=("extreme_rain_day", "sum"),
            max_daily_rainfall_mm=("rainfall_mm", "max"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly_risk["rainy_day_pct"] = monthly_risk["rainy_day_pct"] * 100
    monthly_risk["month_risk_score_0_100"] = (
        percentile_score(monthly_risk["mean_daily_rainfall_mm"])
        + percentile_score(monthly_risk["rainy_day_pct"])
        + percentile_score(monthly_risk["heavy_rain_days"])
    ) / 3

    district_risk = (
        eda.groupby("district")
        .agg(
            total_rainfall_mm=("rainfall_mm", "sum"),
            mean_daily_rainfall_mm=("rainfall_mm", "mean"),
            rainy_day_pct=("rainy_day", "mean"),
            heavy_rain_days=("heavy_rain_day", "sum"),
            very_heavy_rain_days=("very_heavy_rain_day", "sum"),
            extreme_rain_days=("extreme_rain_day", "sum"),
            max_daily_rainfall_mm=("rainfall_mm", "max"),
            records=("rainfall_mm", "size"),
        )
        .reset_index()
    )
    district_risk["rainy_day_pct"] = district_risk["rainy_day_pct"] * 100
    district_risk["avg_annual_rainfall_mm"] = district_risk["district"].map(
        district_year.groupby("district")["annual_rainfall_mm"].mean()
    )
    district_risk["district_risk_score_0_100"] = (
        percentile_score(district_risk["avg_annual_rainfall_mm"])
        + percentile_score(district_risk["rainy_day_pct"])
        + percentile_score(district_risk["heavy_rain_days"])
        + percentile_score(district_risk["max_daily_rainfall_mm"])
    ) / 4
    district_risk = district_risk.sort_values("district_risk_score_0_100", ascending=False)

    district_month = (
        eda.groupby(["district", "month", "month_name"])
        .agg(
            mean_daily_rainfall_mm=("rainfall_mm", "mean"),
            rainy_day_pct=("rainy_day", "mean"),
            heavy_rain_days=("heavy_rain_day", "sum"),
        )
        .reset_index()
    )
    district_month["rainy_day_pct"] = district_month["rainy_day_pct"] * 100

    recent_district_rainfall = (
        recent_eda.groupby("district")
        .agg(
            total_rainfall_last_14_years_mm=("rainfall_mm", "sum"),
            avg_annual_rainfall_last_14_years_mm=("rainfall_mm", lambda values: values.sum() / recent_eda["year"].nunique()),
            rainy_day_pct=("rainy_day", "mean"),
            heavy_rain_days=("heavy_rain_day", "sum"),
            very_heavy_rain_days=("very_heavy_rain_day", "sum"),
            extreme_rain_days=("extreme_rain_day", "sum"),
            max_daily_rainfall_mm=("rainfall_mm", "max"),
        )
        .reset_index()
    )
    recent_district_rainfall["rainy_day_pct"] = recent_district_rainfall["rainy_day_pct"] * 100
    recent_district_rainfall["rank_by_total_rainfall"] = (
        recent_district_rainfall["total_rainfall_last_14_years_mm"].rank(ascending=False, method="dense").astype(int)
    )
    recent_district_rainfall = recent_district_rainfall.sort_values("total_rainfall_last_14_years_mm", ascending=False)

    return {
        "yearly_risk": yearly_risk,
        "monthly_risk": monthly_risk,
        "district_risk": district_risk,
        "district_month_risk": district_month,
        "district_year_risk": district_year,
        "district_rainfall_last_14_years": recent_district_rainfall,
    }


def save_eda_plots(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yearly = tables["yearly_risk"]
    monthly = tables["monthly_risk"]
    district = tables["district_risk"]
    district_month = tables["district_month_risk"]
    recent_district = tables["district_rainfall_last_14_years"]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(yearly["year"], yearly["avg_district_annual_rainfall_mm"], marker="o", color="#2f6f9f")
    ax1.set_title("Yearly Rainfall Risk Trend")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Avg District Annual Rainfall (mm)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(yearly["year"], yearly["heavy_rain_days"], alpha=0.25, color="#d95f02")
    ax2.set_ylabel("Heavy Rain Days")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_yearly_rainfall_risk.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(monthly["month_name"], monthly["mean_daily_rainfall_mm"], color="#4c78a8")
    ax1.set_title("Monthly Rainfall Risk")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean Daily Rainfall (mm)")
    ax2 = ax1.twinx()
    ax2.plot(monthly["month_name"], monthly["rainy_day_pct"], marker="o", color="#f58518")
    ax2.set_ylabel("Rainy Days (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_monthly_rainfall_risk.png", dpi=160)
    plt.close(fig)

    top_districts = district.head(20).sort_values("district_risk_score_0_100")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_districts["district"], top_districts["district_risk_score_0_100"], color="#54a24b")
    ax.set_title("Most Vulnerable Districts for Rainfall")
    ax.set_xlabel("Composite Risk Score (0-100)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_top20_vulnerable_districts.png", dpi=160)
    plt.close(fig)

    heatmap_districts = district.head(20)["district"]
    heatmap = (
        district_month[district_month["district"].isin(heatmap_districts)]
        .pivot(index="district", columns="month", values="mean_daily_rainfall_mm")
        .loc[heatmap_districts]
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(heatmap.values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Rainfall Vulnerability Heatmap: Top Districts by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("District")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    fig.colorbar(image, ax=ax, label="Mean Daily Rainfall (mm)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_district_month_heatmap.png", dpi=160)
    plt.close(fig)

    recent_top = recent_district.head(20).sort_values("total_rainfall_last_14_years_mm")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(recent_top["district"], recent_top["total_rainfall_last_14_years_mm"], color="#2f6f9f")
    ax.set_title("Districts with Highest Total Rainfall in the Last 14 Years")
    ax.set_xlabel("Total Rainfall (mm)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_top20_total_rainfall_last_14_years.png", dpi=160)
    plt.close(fig)


def save_eda_summary(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    yearly = tables["yearly_risk"].sort_values("year_risk_score_0_100", ascending=False).head(5)
    monthly = tables["monthly_risk"].sort_values("month_risk_score_0_100", ascending=False).head(5)
    district = tables["district_risk"].head(10)
    recent_district = tables["district_rainfall_last_14_years"].head(10)

    lines = [
        "Rainfall EDA Summary",
        "",
        "High-risk years:",
        yearly[
            [
                "year",
                "avg_district_annual_rainfall_mm",
                "heavy_rain_days",
                "max_daily_rainfall_mm",
                "year_risk_score_0_100",
            ]
        ].round(2).to_string(index=False),
        "",
        "High-risk months:",
        monthly[
            [
                "month_name",
                "mean_daily_rainfall_mm",
                "rainy_day_pct",
                "heavy_rain_days",
                "month_risk_score_0_100",
            ]
        ].round(2).to_string(index=False),
        "",
        "Most vulnerable districts:",
        district[
            [
                "district",
                "avg_annual_rainfall_mm",
                "rainy_day_pct",
                "heavy_rain_days",
                "max_daily_rainfall_mm",
                "district_risk_score_0_100",
            ]
        ].round(2).to_string(index=False),
        "",
        "Districts with highest total rainfall in the last 14 years:",
        recent_district[
            [
                "rank_by_total_rainfall",
                "district",
                "total_rainfall_last_14_years_mm",
                "avg_annual_rainfall_last_14_years_mm",
                "heavy_rain_days",
                "max_daily_rainfall_mm",
            ]
        ].round(2).to_string(index=False),
        "",
        f"Rain day threshold: > {RAIN_THRESHOLD_MM} mm",
        f"Heavy rain threshold: >= {HEAVY_RAIN_MM} mm",
        f"Very heavy rain threshold: >= {VERY_HEAVY_RAIN_MM} mm",
        f"Extreme rain threshold: >= {EXTREME_RAIN_MM} mm",
    ]
    (output_dir / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_eda(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_weather(Path(args.data), args.rain_cap_quantile)
    tables = build_eda_tables(df)

    for name, table in tables.items():
        table.to_csv(output_dir / f"eda_{name}.csv", index=False)

    save_eda_plots(tables, output_dir)
    save_eda_summary(tables, output_dir)

    print(f"EDA outputs saved to: {output_dir}")
    print("\nTop high-risk months")
    print(
        tables["monthly_risk"]
        .sort_values("month_risk_score_0_100", ascending=False)
        .head(5)
        [["month_name", "mean_daily_rainfall_mm", "rainy_day_pct", "heavy_rain_days", "month_risk_score_0_100"]]
        .round(2)
        .to_string(index=False)
    )
    print("\nTop vulnerable districts")
    print(
        tables["district_risk"]
        .head(10)
        [["district", "avg_annual_rainfall_mm", "rainy_day_pct", "heavy_rain_days", "district_risk_score_0_100"]]
        .round(2)
        .to_string(index=False)
    )
    print("\nDistricts with highest total rainfall in the last 14 years")
    print(
        tables["district_rainfall_last_14_years"]
        .head(10)
        [
            [
                "rank_by_total_rainfall",
                "district",
                "total_rainfall_last_14_years_mm",
                "avg_annual_rainfall_last_14_years_mm",
                "heavy_rain_days",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )


def split_train_validation_test(supervised: pd.DataFrame, train_end_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data = supervised[supervised["year"] < train_end_year]
    validation_data = supervised[supervised["year"] == train_end_year]
    test_data = supervised[supervised["year"] > train_end_year]

    if validation_data.empty:
        candidate_train = supervised[supervised["year"] <= train_end_year].sort_values(["date", "district", "horizon"])
        split_index = int(len(candidate_train) * 0.85)
        train_data = candidate_train.iloc[:split_index]
        validation_data = candidate_train.iloc[split_index:]

    return train_data, validation_data, test_data


def save_xgboost_training_curves(regressor: XGBRegressor, classifier: XGBClassifier, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reg_history = regressor.evals_result()
    cls_history = classifier.evals_result()

    curve_rows = []
    for split_key, split_name in [("validation_0", "train"), ("validation_1", "validation")]:
        for metric, values in reg_history.get(split_key, {}).items():
            curve_rows.extend(
                {"model": "xgboost_regressor", "split": split_name, "metric": metric, "iteration": i + 1, "value": value}
                for i, value in enumerate(values)
            )
        for metric, values in cls_history.get(split_key, {}).items():
            curve_rows.extend(
                {"model": "xgboost_classifier", "split": split_name, "metric": metric, "iteration": i + 1, "value": value}
                for i, value in enumerate(values)
            )
    pd.DataFrame(curve_rows).to_csv(output_dir / "xgboost_training_history.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    for split_key, label in [("validation_0", "Train"), ("validation_1", "Validation")]:
        values = reg_history.get(split_key, {}).get("rmse")
        if values:
            axes[0].plot(values, label=label)
    axes[0].set_title("XGBoost Regression Loss")
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("RMSE on log rainfall")

    for split_key, label in [("validation_0", "Train"), ("validation_1", "Validation")]:
        values = cls_history.get(split_key, {}).get("logloss")
        if values:
            axes[1].plot(values, label=label)
    axes[1].set_title("XGBoost Classification Loss")
    axes[1].set_xlabel("Boosting Round")
    axes[1].set_ylabel("Log Loss")

    for split_key, label in [("validation_0", "Train"), ("validation_1", "Validation")]:
        values = cls_history.get(split_key, {}).get("error")
        if values:
            axes[2].plot([1 - value for value in values], label=label)
    axes[2].set_title("XGBoost Rain/No-Rain Accuracy")
    axes[2].set_xlabel("Boosting Round")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_training_validation_curves.png", dpi=160)
    plt.close(fig)


def lstm_feature_columns() -> list[str]:
    return [
        *RAW_WEATHER_COLUMNS,
        "rainfall_mm",
        "dewpoint_depression",
        "wind_speed",
        "moisture_flux",
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_monsoon",
        "district_enc",
    ]


def build_lstm_arrays(
    df: pd.DataFrame,
    sequence_length: int,
    random_state: int,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features = lstm_feature_columns()
    x_parts = []
    y_reg_parts = []
    y_cls_parts = []
    years = []

    for _, group in df.sort_values(["district", "date"]).groupby("district", sort=False):
        values = group[features].to_numpy(dtype=np.float32)
        rainfall = group["rainfall_mm"].to_numpy(dtype=np.float32)
        group_years = group["year"].to_numpy()
        max_start = len(group) - sequence_length - len(list(HORIZONS)) + 1
        if max_start <= 0:
            continue
        for start in range(max_start):
            end = start + sequence_length
            future = rainfall[end : end + len(list(HORIZONS))]
            x_parts.append(values[start:end])
            y_reg_parts.append(np.log1p(future))
            y_cls_parts.append((future > RAIN_THRESHOLD_MM).astype(np.float32))
            years.append(group_years[end - 1])

    x = np.asarray(x_parts, dtype=np.float32)
    y_reg = np.asarray(y_reg_parts, dtype=np.float32)
    y_cls = np.asarray(y_cls_parts, dtype=np.float32)
    years_array = np.asarray(years)

    if max_samples and len(x) > max_samples:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(x), size=max_samples, replace=False)
        x = x[sample_idx]
        y_reg = y_reg[sample_idx]
        y_cls = y_cls[sample_idx]
        years_array = years_array[sample_idx]

    return x, y_reg, y_cls, years_array, features


def save_lstm_history_plots(history: object, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "lstm_training_history.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for column in ["loss", "val_loss"]:
        if column in history_df:
            axes[0].plot(history_df["epoch"], history_df[column], marker="o", label=column)
    axes[0].set_title("LSTM Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    for column in ["rainfall_loss", "val_rainfall_loss"]:
        if column in history_df:
            axes[1].plot(history_df["epoch"], history_df[column], marker="o", label=column)
    axes[1].set_title("LSTM Rainfall Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE on log rainfall")

    accuracy_columns = [column for column in history_df.columns if "rain_probability" in column and "accuracy" in column]
    for column in accuracy_columns:
        axes[2].plot(history_df["epoch"], history_df[column], marker="o", label=column)
    axes[2].set_title("LSTM Rain/No-Rain Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "lstm_training_validation_curves.png", dpi=160)
    plt.close(fig)


def train_lstm_model(df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Model

    x, y_reg, y_cls, years, features = build_lstm_arrays(
        df=df,
        sequence_length=args.lstm_sequence_length,
        random_state=args.random_state,
        max_samples=args.max_lstm_samples,
    )
    train_mask = years < args.train_end_year
    validation_mask = years == args.train_end_year
    test_mask = years > args.train_end_year

    if not validation_mask.any():
        train_indices = np.flatnonzero(train_mask)
        split = int(len(train_indices) * 0.85)
        validation_indices = train_indices[split:]
        train_indices = train_indices[:split]
    else:
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)

    if len(train_indices) == 0 or len(validation_indices) == 0 or len(test_indices) == 0:
        raise ValueError("LSTM split is empty. Try increasing --max-lstm-samples or checking date coverage.")

    scaler = StandardScaler()
    x_train_2d = x[train_indices].reshape(-1, x.shape[-1])
    scaler.fit(x_train_2d)

    def scale_x(selected: np.ndarray) -> np.ndarray:
        shaped = x[selected].reshape(-1, x.shape[-1])
        scaled = scaler.transform(shaped)
        return scaled.reshape(len(selected), x.shape[1], x.shape[2]).astype(np.float32)

    x_train = scale_x(train_indices)
    x_val = scale_x(validation_indices)
    x_test = scale_x(test_indices)

    inputs = Input(shape=(x.shape[1], x.shape[2]))
    hidden = LSTM(64)(inputs)
    hidden = Dropout(0.2)(hidden)
    hidden = Dense(48, activation="relu")(hidden)
    rainfall = Dense(len(list(HORIZONS)), name="rainfall")(hidden)
    rain_probability = Dense(len(list(HORIZONS)), activation="sigmoid", name="rain_probability")(hidden)
    model = Model(inputs=inputs, outputs=[rainfall, rain_probability])
    model.compile(
        optimizer="adam",
        loss={"rainfall": "mse", "rain_probability": "binary_crossentropy"},
        metrics={"rainfall": ["mae"], "rain_probability": ["accuracy"]},
    )

    print("\nTraining LSTM comparison model...")
    print(f"LSTM train rows: {len(train_indices):,}")
    print(f"LSTM validation rows: {len(validation_indices):,}")
    print(f"LSTM test rows: {len(test_indices):,}")

    history = model.fit(
        x_train,
        {"rainfall": y_reg[train_indices], "rain_probability": y_cls[train_indices]},
        validation_data=(x_val, {"rainfall": y_reg[validation_indices], "rain_probability": y_cls[validation_indices]}),
        epochs=args.lstm_epochs,
        batch_size=args.lstm_batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=1,
    )

    save_lstm_history_plots(history, output_dir)

    pred_log, pred_prob = model.predict(x_test, batch_size=args.lstm_batch_size, verbose=0)
    pred = np.clip(np.expm1(pred_log), 0, None)
    actual = np.expm1(y_reg[test_indices])
    pred_cls = (pred_prob >= 0.5).astype(int)

    rows = []
    for idx, horizon in enumerate(HORIZONS):
        rows.append(
            {
                "model": "LSTM",
                "horizon_day": horizon,
                "mae_mm": mean_absolute_error(actual[:, idx], pred[:, idx]),
                "rmse_mm": np.sqrt(mean_squared_error(actual[:, idx], pred[:, idx])),
                "r2": r2_score(actual[:, idx], pred[:, idx]),
                "rain_accuracy": accuracy_score(y_cls[test_indices][:, idx], pred_cls[:, idx]),
                "rows": len(test_indices),
            }
        )

    lstm_metrics = pd.DataFrame(rows)
    lstm_metrics.to_csv(output_dir / "lstm_metrics_by_horizon.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "mae_mm": mean_absolute_error(actual.reshape(-1), pred.reshape(-1)),
                "rmse_mm": np.sqrt(mean_squared_error(actual.reshape(-1), pred.reshape(-1))),
                "r2": r2_score(actual.reshape(-1), pred.reshape(-1)),
                "rain_accuracy": accuracy_score(y_cls[test_indices].reshape(-1), pred_cls.reshape(-1)),
            }
        ]
    )
    overall.to_csv(output_dir / "lstm_overall_metrics.csv", index=False)

    model.save(output_dir / "rainfall_lstm_model.keras")
    joblib.dump(scaler, output_dir / "lstm_feature_scaler.pkl")
    joblib.dump(features, output_dir / "lstm_feature_columns.pkl")

    print("\nLSTM test metrics by forecast day")
    print(lstm_metrics.round(4).to_string(index=False))
    return overall


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_weather(data_path, args.rain_cap_quantile)
    df, encoder = add_as_of_features(df)
    supervised = make_supervised_7day(df)

    if args.sample_frac is not None:
        supervised = supervised.sample(frac=args.sample_frac, random_state=args.random_state)

    train_data, validation_data, test_data = split_train_validation_test(supervised, args.train_end_year)

    features = feature_columns()
    x_train = train_data[features]
    x_validation = validation_data[features]
    x_test = test_data[features]
    y_train_log = np.log1p(train_data["target_rainfall_mm"])
    y_validation_log = np.log1p(validation_data["target_rainfall_mm"])
    y_train = train_data["target_rainfall_mm"]
    y_validation = validation_data["target_rainfall_mm"]
    y_test = test_data["target_rainfall_mm"]
    y_train_cls = train_data["target_rain_occurred"]
    y_validation_cls = validation_data["target_rain_occurred"]
    y_test_cls = test_data["target_rain_occurred"]

    regressor, classifier = make_models(args.random_state)

    print(f"Training rows: {len(train_data):,}")
    print(f"Validation rows: {len(validation_data):,}")
    print(f"Testing rows : {len(test_data):,}")
    print("Training 7-day rainfall amount model...")
    regressor.fit(
        x_train,
        y_train_log,
        eval_set=[(x_train, y_train_log), (x_validation, y_validation_log)],
        verbose=False,
    )

    print("Training 7-day rain/no-rain probability model...")
    classifier.fit(
        x_train,
        y_train_cls,
        eval_set=[(x_train, y_train_cls), (x_validation, y_validation_cls)],
        verbose=False,
    )

    train_pred = np.expm1(regressor.predict(x_train))
    train_pred = np.clip(train_pred, 0, None)
    validation_pred = np.expm1(regressor.predict(x_validation))
    validation_pred = np.clip(validation_pred, 0, None)
    test_pred = np.expm1(regressor.predict(x_test))
    test_pred = np.clip(test_pred, 0, None)
    train_cls_pred = classifier.predict(x_train)
    validation_cls_pred = classifier.predict(x_validation)
    test_cls_pred = classifier.predict(x_test)
    test_proba = classifier.predict_proba(x_test)[:, 1]

    overall = pd.DataFrame(
        [
            {
                "split": "train",
                "mae_mm": mean_absolute_error(y_train, train_pred),
                "rmse_mm": np.sqrt(mean_squared_error(y_train, train_pred)),
                "r2": r2_score(y_train, train_pred),
                "rain_accuracy": accuracy_score(y_train_cls, train_cls_pred),
            },
            {
                "split": "validation",
                "mae_mm": mean_absolute_error(y_validation, validation_pred),
                "rmse_mm": np.sqrt(mean_squared_error(y_validation, validation_pred)),
                "r2": r2_score(y_validation, validation_pred),
                "rain_accuracy": accuracy_score(y_validation_cls, validation_cls_pred),
            },
            {
                "split": "test",
                "mae_mm": mean_absolute_error(y_test, test_pred),
                "rmse_mm": np.sqrt(mean_squared_error(y_test, test_pred)),
                "r2": r2_score(y_test, test_pred),
                "rain_accuracy": accuracy_score(y_test_cls, test_cls_pred),
            },
        ]
    )
    regression_metrics = pd.concat(
        [
            evaluate_regression_by_horizon(train_data, train_pred, "train"),
            evaluate_regression_by_horizon(validation_data, validation_pred, "validation"),
            evaluate_regression_by_horizon(test_data, test_pred, "test"),
        ],
        ignore_index=True,
    )
    classification_metrics = pd.concat(
        [
            evaluate_classification_by_horizon(train_data, train_cls_pred, "train"),
            evaluate_classification_by_horizon(validation_data, validation_cls_pred, "validation"),
            evaluate_classification_by_horizon(test_data, test_cls_pred, "test"),
        ],
        ignore_index=True,
    )

    overall.to_csv(output_dir / "7day_overall_train_validation_test_metrics.csv", index=False)
    overall.to_csv(output_dir / "7day_overall_train_test_metrics.csv", index=False)
    regression_metrics.to_csv(output_dir / "7day_regression_metrics_by_horizon.csv", index=False)
    classification_metrics.to_csv(output_dir / "7day_classification_metrics_by_horizon.csv", index=False)
    save_evaluation_plots(regression_metrics, classification_metrics, test_data, test_pred, output_dir)
    save_xgboost_training_curves(regressor, classifier, output_dir)

    print("\nOverall train/test metrics")
    print(overall.round(4).to_string(index=False))

    print("\nRegression metrics by forecast day")
    print(regression_metrics.round(4).to_string(index=False))

    print("\nClassification metrics by forecast day")
    print(classification_metrics.round(4).to_string(index=False))

    forecast_preview = test_data[["district", "date", "target_date", "horizon", "target_rainfall_mm"]].copy()
    forecast_preview["predicted_rainfall_mm"] = test_pred
    forecast_preview["rain_probability"] = test_proba
    forecast_preview["rain_prediction"] = np.where(test_cls_pred == 1, "Rain", "No Rain")
    forecast_preview.to_csv(output_dir / "7day_test_predictions.csv", index=False)

    xgb_comparison = overall[overall["split"] == "test"].copy()
    xgb_comparison.insert(0, "model", "XGBoost")
    xgb_comparison = xgb_comparison.drop(columns=["split"])

    comparison = xgb_comparison
    if args.include_lstm:
        lstm_comparison = train_lstm_model(df, args, output_dir)
        comparison = pd.concat([xgb_comparison, lstm_comparison], ignore_index=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print("\nModel comparison on test data")
    print(comparison.round(4).to_string(index=False))

    if not args.skip_save:
        joblib.dump(regressor, output_dir / "rainfall_7day_regressor.pkl")
        joblib.dump(classifier, output_dir / "rainfall_7day_classifier.pkl")
        joblib.dump(encoder, output_dir / "district_encoder.pkl")
        joblib.dump(features, output_dir / "feature_columns.pkl")
        print(f"\nSaved models and artifacts to: {output_dir}")


def forecast_next_7_days(
    data_path: Path,
    output_dir: Path,
    district: str,
    as_of_date: str | None = None,
    rain_cap_quantile: float = 0.99,
    rain_probability_threshold: float = 0.5,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    regressor = joblib.load(output_dir / "rainfall_7day_regressor.pkl")
    classifier = joblib.load(output_dir / "rainfall_7day_classifier.pkl")
    encoder = joblib.load(output_dir / "district_encoder.pkl")
    features = joblib.load(output_dir / "feature_columns.pkl")

    df = load_weather(Path(data_path), rain_cap_quantile)
    df, _ = add_as_of_features(df, encoder)

    district_rows = df[df["district"].str.lower() == district.lower()].sort_values("date")
    if district_rows.empty:
        available = ", ".join(sorted(df["district"].unique())[:10])
        raise ValueError(f"District not found. Examples available: {available}")

    as_of_date = pd.to_datetime(as_of_date) if as_of_date else district_rows["date"].max() - pd.Timedelta(days=7)
    current = district_rows[district_rows["date"] == as_of_date]
    if current.empty:
        raise ValueError(f"No row found for {district} on {as_of_date.date()}")

    forecast_rows = []
    for horizon in HORIZONS:
        target_date = as_of_date + pd.Timedelta(days=horizon)
        target_weather = district_rows[district_rows["date"] == target_date]
        if target_weather.empty:
            raise ValueError(f"No weather row found for {district} on forecast date {target_date.date()}")

        row = target_weather.iloc[[0]].copy()
        current_row = current.iloc[0]
        for lag in LAGS:
            row[f"rain_lag_{lag}"] = current_row[f"rain_lag_{lag}"]
        for window in ROLL_WINDOWS:
            row[f"rain_roll{window}_mean"] = current_row[f"rain_roll{window}_mean"]
            row[f"rain_roll{window}_sum"] = current_row[f"rain_roll{window}_sum"]
        row["district_enc"] = current_row["district_enc"]
        row["horizon"] = horizon
        row["target_date"] = target_date

        row["target_month_sin"] = np.sin(2 * np.pi * target_date.month / 12)
        row["target_month_cos"] = np.cos(2 * np.pi * target_date.month / 12)
        row["target_dayofyear_sin"] = np.sin(2 * np.pi * target_date.dayofyear / 366)
        row["target_dayofyear_cos"] = np.cos(2 * np.pi * target_date.dayofyear / 366)
        forecast_rows.append(row)

    forecast = pd.concat(forecast_rows, ignore_index=True)
    pred = np.expm1(regressor.predict(forecast[features]))
    pred = np.clip(pred, 0, None)
    proba = classifier.predict_proba(forecast[features])[:, 1]

    return pd.DataFrame(
        {
            "district": forecast["district"],
            "as_of_date": as_of_date.date(),
            "forecast_date": forecast["target_date"].dt.date,
            "day_ahead": forecast["horizon"],
            "predicted_rainfall_mm": pred,
            "rain_probability": proba,
            "rain_prediction": np.where(proba >= rain_probability_threshold, "Rain", "No Rain"),
        }
    )


def read_json_url(url: str, timeout_seconds: int = 30) -> dict:
    with urlopen(url, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode_place(place: str, country_code: str = "IN") -> dict:
    params = {
        "name": place,
        "count": 5,
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["countryCode"] = country_code
    data = read_json_url(f"{OPEN_METEO_GEOCODING_URL}?{urlencode(params)}")
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not find location: {place}")
    return results[0]


def normalize_location_name(name: str) -> str:
    normalized = name.lower().replace(" district", "").replace("division", "")
    return " ".join(normalized.replace("-", " ").split())


def match_model_district(location: dict, known_districts: Iterable[str]) -> str:
    known = list(known_districts)
    known_lookup = {normalize_location_name(name): name for name in known}
    candidates = [
        location.get("admin2", ""),
        location.get("admin3", ""),
        location.get("admin1", ""),
        location.get("name", ""),
    ]

    for candidate in candidates:
        normalized = normalize_location_name(candidate)
        if normalized in known_lookup:
            return known_lookup[normalized]

    all_candidates = [normalize_location_name(candidate) for candidate in candidates if candidate]
    matches = []
    for candidate in all_candidates:
        matches.extend(difflib.get_close_matches(candidate, known_lookup.keys(), n=1, cutoff=0.72))

    if matches:
        return known_lookup[matches[0]]

    examples = ", ".join(known[:10])
    location_name = location.get("name", "this location")
    raise ValueError(
        f"Could not map {location_name} to a district used by the trained model. "
        f"Try a Uttar Pradesh district name. Examples: {examples}"
    )


def fetch_open_meteo_hourly(latitude: float, longitude: float, past_days: int = 35, forecast_days: int = 8) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
        "daily": ",".join(OPEN_METEO_DAILY_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    data = read_json_url(f"{OPEN_METEO_FORECAST_URL}?{urlencode(params)}")
    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo did not return hourly weather data.")

    weather = pd.DataFrame(hourly)
    weather["time"] = pd.to_datetime(weather["time"])
    weather["date"] = weather["time"].dt.normalize()

    for column in OPEN_METEO_HOURLY_VARIABLES:
        if column not in weather.columns:
            weather[column] = np.nan

    direction_rad = np.deg2rad(weather["wind_direction_10m"].fillna(0))
    wind_speed = weather["wind_speed_10m"].fillna(0)
    weather["u10_component"] = -wind_speed * np.sin(direction_rad)
    weather["v10_component"] = -wind_speed * np.cos(direction_rad)
    water_vapour = weather["total_column_integrated_water_vapour"].fillna(0)
    weather["viwve_proxy"] = water_vapour * weather["u10_component"]
    weather["viwvn_proxy"] = water_vapour * weather["v10_component"]

    daily = (
        weather.groupby("date")
        .agg(
            sp=("surface_pressure", "mean"),
            tcc=("cloud_cover", "mean"),
            u10=("u10_component", "mean"),
            v10=("v10_component", "mean"),
            t2m=("temperature_2m", "mean"),
            d2m=("dew_point_2m", "mean"),
            relative_humidity_2m=("relative_humidity_2m", "mean"),
            lcc=("cloud_cover_low", "mean"),
            viwve=("viwve_proxy", "mean"),
            viwvn=("viwvn_proxy", "mean"),
            rainfall_mm=("precipitation", "sum"),
        )
        .reset_index()
        .rename(columns={"date": "date"})
    )
    daily["sp"] = daily["sp"] * 100.0
    daily["tcc"] = daily["tcc"] / 100.0
    daily["lcc"] = daily["lcc"] / 100.0

    daily_forecast = data.get("daily") or {}
    if daily_forecast and "time" in daily_forecast:
        daily_display = pd.DataFrame(daily_forecast)
        daily_display["date"] = pd.to_datetime(daily_display["time"]).dt.normalize()
        daily_display = daily_display.drop(columns=["time"])
        daily = daily.merge(daily_display, on="date", how="left")

    return daily


def get_value_by_date(df: pd.DataFrame, date: pd.Timestamp, column: str, default: float = 0.0) -> float:
    value = df.loc[df["date"] == date, column]
    if value.empty or pd.isna(value.iloc[0]):
        return default
    return float(value.iloc[0])


def build_live_feature_rows(
    daily_weather: pd.DataFrame,
    district: str,
    district_enc: int,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    daily = daily_weather.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.sort_values("date").reset_index(drop=True)

    requested_as_of = pd.to_datetime(as_of_date).normalize() if as_of_date else pd.Timestamp.today().normalize()
    available_past = daily[daily["date"] <= requested_as_of]
    if available_past.empty:
        raise ValueError("No current or past weather rows are available from the live API.")

    base_date = available_past["date"].max()
    history = daily[daily["date"] < base_date].copy()
    if len(history) < max(ROLL_WINDOWS):
        raise ValueError(f"Need at least {max(ROLL_WINDOWS)} past days from the live API to build lag features.")

    lag_values = {f"rain_lag_{lag}": get_value_by_date(daily, base_date - pd.Timedelta(days=lag), "rainfall_mm") for lag in LAGS}
    roll_values = {}
    for window in ROLL_WINDOWS:
        window_values = history.tail(window)["rainfall_mm"]
        roll_values[f"rain_roll{window}_mean"] = float(window_values.mean())
        roll_values[f"rain_roll{window}_sum"] = float(window_values.sum())

    rows = []
    for horizon in HORIZONS:
        forecast_date = base_date + pd.Timedelta(days=horizon)
        weather_row = daily[daily["date"] == forecast_date]
        if weather_row.empty:
            raise ValueError(f"Open-Meteo did not return weather for forecast date {forecast_date.date()}")

        row = weather_row.iloc[0].copy()
        row["district"] = district
        row["district_enc"] = district_enc
        row["year"] = base_date.year
        row["horizon"] = horizon
        row["target_date"] = forecast_date
        row["target_month_sin"] = np.sin(2 * np.pi * forecast_date.month / 12)
        row["target_month_cos"] = np.cos(2 * np.pi * forecast_date.month / 12)
        row["target_dayofyear_sin"] = np.sin(2 * np.pi * forecast_date.dayofyear / 366)
        row["target_dayofyear_cos"] = np.cos(2 * np.pi * forecast_date.dayofyear / 366)

        row["month"] = forecast_date.month
        row["dayofyear"] = forecast_date.dayofyear
        row["dewpoint_depression"] = row["t2m"] - row["d2m"]
        row["wind_speed"] = np.sqrt(row["u10"] ** 2 + row["v10"] ** 2)
        row["moisture_flux"] = np.sqrt(row["viwve"] ** 2 + row["viwvn"] ** 2)
        row["is_monsoon"] = int(6 <= forecast_date.month <= 9)
        row["month_sin"] = np.sin(2 * np.pi * forecast_date.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * forecast_date.month / 12)
        row["dayofyear_sin"] = np.sin(2 * np.pi * forecast_date.dayofyear / 366)
        row["dayofyear_cos"] = np.cos(2 * np.pi * forecast_date.dayofyear / 366)

        for name, value in lag_values.items():
            row[name] = value
        for name, value in roll_values.items():
            row[name] = value

        rows.append(row)

    return pd.DataFrame(rows)


def save_prediction_plot(forecast: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = forecast.copy()
    plot_data["forecast_date"] = pd.to_datetime(plot_data["forecast_date"])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(plot_data["forecast_date"].dt.strftime("%d %b"), plot_data["predicted_rainfall_mm"], color="#4c78a8")
    ax1.set_title(title)
    ax1.set_xlabel("Forecast Date")
    ax1.set_ylabel("Predicted Rainfall (mm)")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(plot_data["forecast_date"].dt.strftime("%d %b"), plot_data["rain_probability"] * 100, marker="o", color="#f58518")
    ax2.set_ylabel("Rain Probability (%)")
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def forecast_live_next_7_days(
    output_dir: Path,
    place: str,
    country_code: str = "IN",
    as_of_date: str | None = None,
    rain_probability_threshold: float = 0.5,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    regressor = joblib.load(output_dir / "rainfall_7day_regressor.pkl")
    classifier = joblib.load(output_dir / "rainfall_7day_classifier.pkl")
    encoder = joblib.load(output_dir / "district_encoder.pkl")
    features = joblib.load(output_dir / "feature_columns.pkl")

    location = geocode_place(place, country_code=country_code)
    model_district = match_model_district(location, encoder.classes_)
    district_enc = int(encoder.transform([model_district])[0])

    daily_weather = fetch_open_meteo_hourly(location["latitude"], location["longitude"])
    forecast_features = build_live_feature_rows(daily_weather, model_district, district_enc, as_of_date=as_of_date)

    pred = np.expm1(regressor.predict(forecast_features[features]))
    pred = np.clip(pred, 0, None)
    proba = classifier.predict_proba(forecast_features[features])[:, 1]

    result = pd.DataFrame(
        {
            "place": place,
            "matched_model_district": model_district,
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "as_of_date": pd.to_datetime(forecast_features["target_date"]) - pd.to_timedelta(forecast_features["horizon"], unit="D"),
            "forecast_date": pd.to_datetime(forecast_features["target_date"]).dt.date,
            "day_ahead": forecast_features["horizon"],
            "predicted_rainfall_mm": pred,
            "rain_probability": proba,
            "rain_prediction": np.where(proba >= rain_probability_threshold, "Rain", "No Rain"),
            "temperature_c": forecast_features["t2m"],
            "temperature_max_c": forecast_features.get("temperature_2m_max", forecast_features["t2m"]),
            "temperature_min_c": forecast_features.get("temperature_2m_min", forecast_features["t2m"]),
            "temperature_mean_c": forecast_features.get("temperature_2m_mean", forecast_features["t2m"]),
            "humidity_pct": forecast_features["relative_humidity_2m"],
            "wind_speed_ms": forecast_features["wind_speed"],
            "wind_speed_max_ms": forecast_features.get("wind_speed_10m_max", forecast_features["wind_speed"]),
            "wind_gusts_max_ms": forecast_features.get("wind_gusts_10m_max", forecast_features["wind_speed"]),
            "surface_pressure_pa": forecast_features["sp"],
            "cloud_cover_pct": forecast_features["tcc"] * 100,
            "open_meteo_precipitation_probability_pct": forecast_features.get("precipitation_probability_max", np.nan),
        }
    )
    result["as_of_date"] = pd.to_datetime(result["as_of_date"]).dt.date
    return result


def live_predict(args: argparse.Namespace) -> None:
    result = forecast_live_next_7_days(
        output_dir=Path(args.output_dir),
        place=args.place,
        country_code=args.country_code,
        as_of_date=args.as_of_date,
        rain_probability_threshold=args.rain_probability_threshold,
    )
    print(result.round({"predicted_rainfall_mm": 2, "rain_probability": 3}).to_string(index=False))

    if args.plot_path:
        plot_path = Path(args.plot_path)
    else:
        safe_place = "".join(char if char.isalnum() else "_" for char in args.place).strip("_").lower()
        plot_path = Path(args.output_dir) / f"live_7day_prediction_{safe_place}.png"
    save_prediction_plot(result, plot_path, f"Next 7 Days Rainfall Forecast: {args.place}")
    print(f"\nPrediction graph saved to: {plot_path}")


def predict(args: argparse.Namespace) -> None:
    result = forecast_next_7_days(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        district=args.district,
        as_of_date=args.as_of_date,
        rain_cap_quantile=args.rain_cap_quantile,
        rain_probability_threshold=args.rain_probability_threshold,
    )
    print(result.round({"predicted_rainfall_mm": 2, "rain_probability": 3}).to_string(index=False))

    if args.plot_path:
        save_prediction_plot(result, Path(args.plot_path), f"Next 7 Days Rainfall Forecast: {args.district}")
        print(f"\nPrediction graph saved to: {args.plot_path}")


def add_common_args(parser: argparse.ArgumentParser, output_dir_default: str = "models_7day") -> None:
    parser.add_argument("--data", default="up_daily_weather_dataset.csv", help="Path to the daily weather CSV.")
    parser.add_argument("--output-dir", default=output_dir_default, help="Directory for saved outputs.")
    parser.add_argument("--rain-cap-quantile", type=float, default=0.99, help="Clip rainfall above this quantile. Use 0 to disable.")
    parser.add_argument("--train-end-year", type=int, default=2022, help="Last year used for training.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional training/debug sample fraction.")
    parser.add_argument("--skip-save", action="store_true", help="Do not save model artifacts.")
    parser.add_argument("--include-lstm", action="store_true", help="Train an LSTM comparison model after XGBoost.")
    parser.add_argument("--lstm-epochs", type=int, default=10, help="Maximum LSTM training epochs.")
    parser.add_argument("--lstm-batch-size", type=int, default=256, help="LSTM batch size.")
    parser.add_argument("--lstm-sequence-length", type=int, default=30, help="Number of past days used by LSTM.")
    parser.add_argument("--max-lstm-samples", type=int, default=120000, help="Maximum sequence samples for LSTM training/evaluation.")


def parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)
    eda_common = argparse.ArgumentParser(add_help=False)
    add_common_args(eda_common, output_dir_default="eda_outputs")

    parser = argparse.ArgumentParser(
        description="Train or run a 7-day rainfall forecast model.",
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", parents=[common], help="Train and evaluate the 7-day model.")
    subparsers.add_parser("eda", parents=[eda_common], help="Run exploratory rainfall risk analysis.")

    predict_parser = subparsers.add_parser("predict", parents=[common], help="Predict next 7 days for one district.")
    predict_parser.add_argument("--district", required=True, help="District name, for example Agra.")
    predict_parser.add_argument("--as-of-date", default=None, help="Forecast from this date. Defaults to latest date in the CSV.")
    predict_parser.add_argument("--rain-probability-threshold", type=float, default=0.5, help="Probability cutoff for Rain vs No Rain.")
    predict_parser.add_argument("--plot-path", default=None, help="Optional path to save a 7-day prediction graph.")

    live_parser = subparsers.add_parser("live-predict", parents=[common], help="Fetch live Open-Meteo data and predict next 7 days.")
    live_parser.add_argument("--place", required=True, help="Place or district name, for example Agra or Lucknow.")
    live_parser.add_argument("--country-code", default="IN", help="ISO country code used for geocoding. Defaults to IN.")
    live_parser.add_argument("--as-of-date", default=None, help="Optional as-of date. Defaults to today's live API date.")
    live_parser.add_argument("--rain-probability-threshold", type=float, default=0.5, help="Probability cutoff for Rain vs No Rain.")
    live_parser.add_argument("--plot-path", default=None, help="Optional path to save a 7-day prediction graph.")

    args = parser.parse_args()
    if args.command is None:
        args.command = "train"
    return args


def main() -> None:
    args = parse_args()
    if args.command == "predict":
        predict(args)
    elif args.command == "live-predict":
        live_predict(args)
    elif args.command == "eda":
        run_eda(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
