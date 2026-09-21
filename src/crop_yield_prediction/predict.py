"""Public prediction functions for the AgroAqkol backend/mobile app."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data.akmola_sources import (
    DEFAULT_END_MONTH,
    DEFAULT_START_DAY,
    DEFAULT_START_MONTH,
    FieldInput,
    aggregate_decades,
    fetch_nasa_power_daily,
    fetch_open_meteo_daily,
    fetch_soilgrids,
    make_decade_features,
    precipitation_anomaly,
)
from .risk import compute_risk

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts/models/best_model.joblib"

# stat.gov.kz gov. export for the season used at training time ends 15 Oct
# (see akmola_sources.build_training_dataset) -> every "completed season" row
# in training only ever has decades up to that date, never later.
SEASON_END_MONTH = DEFAULT_END_MONTH  # 10
SEASON_END_DAY = 15

# Approximate Akmola-region topsoil (0-5cm) defaults, used only if the live
# SoilGrids request fails, so the model always gets these columns instead of
# them silently disappearing from the feature row.
_SOIL_DEFAULTS = {
    "soil_clay_0_5cm": 280.0,
    "soil_sand_0_5cm": 420.0,
    "soil_silt_0_5cm": 300.0,
    "soil_soc_0_5cm": 180.0,
    "soil_phh2o_0_5cm": 72.0,
}

_CLIMATE_NORMAL_CACHE: dict[tuple[float, float], pd.DataFrame] = {}


def _load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run scripts/train_model.py first.")
    obj = joblib.load(MODEL_PATH)
    return obj if isinstance(obj, dict) else {"model": obj, "residuals": np.array([]), "model_name": "unknown"}


GOV_CSV_PATH = ROOT / "data/processed/akmola_yield.csv"


def _lag_features(district: str, crop: str, year: int, lag_years: int = 3) -> dict[str, float]:
    """Replicate build_training_dataset's yield_lag_N features for live inference."""
    if not GOV_CSV_PATH.exists():
        return {f"yield_lag_{lag}": np.nan for lag in range(1, lag_years + 1)}
    gov = pd.read_csv(GOV_CSV_PATH)
    gov["district"] = gov["district"].astype(str).str.strip()
    gov["crop"] = gov["crop"].astype(str).str.strip()
    feats: dict[str, float] = {}
    for lag in range(1, lag_years + 1):
        row = gov[(gov["year"] == year - lag) & (gov["district"] == district) & (gov["crop"] == crop)]
        if not row.empty and pd.notna(row.iloc[0]["yield_t_ha"]):
            feats[f"yield_lag_{lag}"] = float(row.iloc[0]["yield_t_ha"])
        else:
            feats[f"yield_lag_{lag}"] = np.nan
    return feats


def _full_decade_grid(year: int) -> list[tuple[int, int]]:
    """Every (month, decade) pair that exists between 1 May and 15 Oct of a season -
    the exact same window used at training time (build_training_dataset) and by
    aggregate_decades. Deriving it from real calendar days (rather than a hardcoded
    list) guarantees it always matches the columns the model was actually fit on."""
    days = pd.date_range(date(year, DEFAULT_START_MONTH, DEFAULT_START_DAY),
                          date(year, SEASON_END_MONTH, SEASON_END_DAY), freq="D")
    decade = np.where(days.day <= 10, 1, np.where(days.day <= 20, 2, 3))
    return sorted(set(zip(days.month, decade)))


def _climate_normal_decades(lat: float, lon: float, history_years: int = 5) -> pd.DataFrame:
    """Average decade weather features over the past `history_years` *completed*
    seasons. Used to fill in decades of the current season that haven't happened yet
    (or aren't finished yet), so the live feature row always has the same decade
    columns the model was trained on, instead of missing them and crashing."""
    key = (round(lat, 3), round(lon, 3))
    if key in _CLIMATE_NORMAL_CACHE:
        return _CLIMATE_NORMAL_CACHE[key]
    this_year = pd.Timestamp.now().year
    frames = []
    for y in range(this_year - history_years, this_year):
        try:
            start = date(y, DEFAULT_START_MONTH, DEFAULT_START_DAY)
            end = date(y, SEASON_END_MONTH, SEASON_END_DAY)
            w = fetch_open_meteo_daily(lat, lon, start.isoformat(), end.isoformat())
            n = fetch_nasa_power_daily(lat, lon, start.isoformat(), end.isoformat())
            frames.append(aggregate_decades(w, n))
        except Exception:
            continue
    if not frames:
        normal = pd.DataFrame(columns=["month", "decade"])
    else:
        all_hist = pd.concat(frames, ignore_index=True)
        metric_cols = [c for c in ["temp_min", "temp_mean", "temp_max", "precipitation_mm",
                                    "humidity_mean", "wind_mean_ms", "wind_max_ms",
                                    "sukhovei_fraction", "gdd_base5", "gwetroot"] if c in all_hist.columns]
        normal = all_hist.groupby(["month", "decade"], as_index=False)[metric_cols].mean()
        normal["precip_anomaly_pct"] = 0.0  # by definition, the "normal" itself has no anomaly
        normal["precip_anomaly_z"] = 0.0
    _CLIMATE_NORMAL_CACHE[key] = normal
    return normal


def _fill_missing_decades(decades: pd.DataFrame, year: int, lat: float, lon: float) -> pd.DataFrame:
    have = set(zip(decades["month"].astype(int), decades["decade"].astype(int))) if not decades.empty else set()
    missing = [md for md in _full_decade_grid(year) if md not in have]
    if not missing:
        return decades
    normal = _climate_normal_decades(lat, lon)
    if normal.empty:
        return decades
    norm_idx = normal.set_index(["month", "decade"])
    fill_rows = []
    for m, d in missing:
        if (m, d) in norm_idx.index:
            r = norm_idx.loc[(m, d)].to_dict()
            r.update({"month": m, "decade": d, "year": year})
            fill_rows.append(r)
    if fill_rows:
        decades = pd.concat([decades, pd.DataFrame(fill_rows)], ignore_index=True)
    return decades


def _field_features(field: FieldInput, season: int | None = None) -> pd.DataFrame:
    year = season or pd.Timestamp.now().year
    is_current_season = year == pd.Timestamp.now().year
    if is_current_season:
        end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = f"{year}-{SEASON_END_MONTH:02d}-{SEASON_END_DAY:02d}"
    weather = fetch_open_meteo_daily(field.latitude, field.longitude, f"{year}-05-01", end_date)
    nasa = fetch_nasa_power_daily(field.latitude, field.longitude, f"{year}-05-01", end_date)
    decades = precipitation_anomaly(aggregate_decades(weather, nasa))

    if is_current_season:
        decades = _fill_missing_decades(decades, year, field.latitude, field.longitude)

    feats: dict[str, Any] = {
        "year": year, "district": field.district, "crop": field.crop,
        "latitude": field.latitude, "longitude": field.longitude, "area_ha": field.area_ha,
    }
    feats.update(make_decade_features(decades))
    feats.update(_lag_features(field.district, field.crop, year))
    try:
        feats.update(fetch_soilgrids(field.latitude, field.longitude))
    except Exception:
        feats.update(_SOIL_DEFAULTS)
    return pd.DataFrame([feats])

_MONTH_RU = {5: "май", 6: "июнь", 7: "июль", 8: "август", 9: "сентябрь", 10: "октябрь"}
_METRIC_RU = {
    "temp_min": "мин. температура",
    "temp_mean": "средняя температура",
    "temp_max": "макс. температура",
    "precipitation_mm": "осадки",
    "humidity_mean": "влажность воздуха",
    "wind_mean_ms": "скорость ветра",
    "wind_max_ms": "макс. скорость ветра",
    "sukhovei_fraction": "доля суховейных дней",
    "gdd_base5": "сумма эффективных температур",
    "gwetroot": "влажность почвы",
    "precip_anomaly_pct": "аномалия осадков",
    "precip_anomaly_z": "аномалия осадков (z-score)",
}
_SOIL_RU = {"clay": "глина", "sand": "песок", "silt": "ил", "soc": "органика почвы", "phh2o": "pH почвы"}


def _humanize_feature_name(raw_name: str) -> str:
    """Translate a raw preprocessed pipeline column (e.g. 'num__weather_08_d2_precipitation_mm')
    into a short human-readable Russian label for the API/UI (top_factors.name)."""
    name = raw_name.split("__", 1)[-1]

    if name.startswith("district_"):
        return f"Район: {name.removeprefix('district_')}"
    if name.startswith("crop_"):
        return f"Культура: {name.removeprefix('crop_')}"

    m = re.match(r"weather_(\d{2})_d(\d)_(.+)", name)
    if m:
        month, decade, metric = int(m.group(1)), m.group(2), m.group(3)
        month_ru = _MONTH_RU.get(month, str(month))
        metric_ru = _METRIC_RU.get(metric, metric)
        return f"{metric_ru}, {month_ru}, {decade}-я декада"

    if name.startswith("soil_"):
        m2 = re.match(r"soil_([a-z0-9]+)_", name)
        if m2:
            return f"Почва: {_SOIL_RU.get(m2.group(1), m2.group(1))}"

    m3 = re.match(r"yield_lag_(\d)", name)
    if m3:
        n = int(m3.group(1))
        year_word = "год" if n == 1 else "года"
        return f"Урожайность {n} {year_word} назад"

    if name in ("year", "latitude", "longitude", "area_ha"):
        return {"year": "год", "latitude": "широта", "longitude": "долгота", "area_ha": "площадь поля"}[name]

    return name


def get_top_factors(field_features: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    """Return top SHAP factors as name/impact/direction."""
    artifact = _load_artifact()
    model = artifact["model"]
    try:
        import shap
        prep = model.named_steps["prep"]
        estimator = model.named_steps["model"]
        transformed = prep.transform(field_features)
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        values = np.asarray(values)[0]
        names = prep.get_feature_names_out()
        idx = np.argsort(np.abs(values))[::-1][:n]
        return [{"name": _humanize_feature_name(str(names[i])), "impact": round(float(values[i]), 3),
                 "direction": "up" if values[i] > 0 else "down"} for i in idx]
    except Exception:
        return []


def predict_yield(field: FieldInput) -> dict[str, Any]:
    """Predict yield t/ha with a bootstrap residual interval and SHAP factors."""
    artifact = _load_artifact()
    model = artifact["model"]
    residuals = np.asarray(artifact.get("residuals", []), dtype=float)
    features = _field_features(field)
    prediction = float(model.predict(features)[0])
    if residuals.size:
        low_r, high_r = np.quantile(residuals, [0.10, 0.90])
        low, high = max(0.0, prediction + low_r), max(0.0, prediction + high_r)
    else:
        low, high = prediction, prediction
    return {
        "yield_t_ha": round(max(0.0, prediction), 3),
        "confidence_interval": [round(float(low), 3), round(float(high), 3)],
        "top_factors": get_top_factors(features, n=3),
        "season": str(int(features.iloc[0]["year"])),
    }


__all__ = ["FieldInput", "predict_yield", "compute_risk", "get_top_factors"]
