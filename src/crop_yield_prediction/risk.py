"""Rule-based decade risk index for drought, sukhovei and early frost/snow."""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .data.akmola_sources import (
    FieldInput,
    aggregate_decades,
    fetch_nasa_power_daily,
    fetch_open_meteo_daily,
    precipitation_anomaly,
)


def _clip1(x: float) -> float:
    """Clip a 0-100 raw score into the 0.0-1.0 range used by the API contract."""
    return float(np.clip(x, 0, 100)) / 100.0


def _drought_component(row: pd.Series) -> float:
    """Drought score (0.0-1.0): precipitation deficit + low root-zone wetness."""
    deficit = max(0.0, -float(row.get("precip_anomaly_pct", 0.0)))
    precip_score = min(100.0, deficit * 1.25)
    gw = row.get("gwetroot", np.nan)
    wetness_score = 0.0 if pd.isna(gw) else float(np.clip((0.50 - float(gw)) / 0.50 * 100, 0, 100))
    return _clip1(max(precip_score, wetness_score))


def _frost_scores(weather: pd.DataFrame, history: list[pd.DataFrame]) -> dict[str, float]:
    """Return frost score by current first-freeze timing versus historical median."""
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    current_freezes = w.loc[w["temperature_2m_min"] < 0, "date"]
    current_doy = int(current_freezes.iloc[0].dayofyear) if not current_freezes.empty else None
    historical_doys: list[int] = []
    for h in history:
        h = h.copy()
        h["date"] = pd.to_datetime(h["date"])
        f = h.loc[h["temperature_2m_min"] < 0, "date"]
        if not f.empty:
            historical_doys.append(int(f.iloc[0].dayofyear))
    if current_doy is None or len(historical_doys) < 2:
        return {"frost": 0.0, "first_frost_doy": float(current_doy or np.nan), "frost_median_doy": float(np.median(historical_doys)) if historical_doys else np.nan}
    median = float(np.median(historical_doys))
    # 14+ days earlier than normal => 1.0, linearly scaled.
    score = _clip1((median - current_doy) / 14.0 * 100)
    return {"frost": score, "first_frost_doy": float(current_doy), "frost_median_doy": median}


def compute_risk(field: FieldInput, year: int | None = None, history_years: int = 10,
                 start_month: int = 5, end_month: int = 10) -> dict[str, Any]:
    """Compute decade risk 0.0-1.0 for one field (API contract: 0-1, not 0-100).

    The final decade score is the maximum of drought, sukhovei and frost components,
    with a human-readable note and an overall alert flag.
    """
    target_year = year or pd.Timestamp.now().year
    start = date(target_year, start_month, 1)
    if target_year == pd.Timestamp.now().year:
        end = (pd.Timestamp.now() - pd.Timedelta(days=1)).date()
    else:
        end = date(target_year, end_month, 15)
    weather = fetch_open_meteo_daily(field.latitude, field.longitude, start.isoformat(), end.isoformat())
    nasa = fetch_nasa_power_daily(field.latitude, field.longitude, start.isoformat(), end.isoformat())

    history: list[pd.DataFrame] = []
    for y in range(target_year - history_years, target_year):
        try:
            h = fetch_open_meteo_daily(field.latitude, field.longitude,
                                       date(y, start_month, 1).isoformat(), date(y, end_month, 15).isoformat())
            history.append(h)
        except Exception:
            continue

    decades = precipitation_anomaly(aggregate_decades(weather, nasa),
                                    pd.concat([aggregate_decades(h) for h in history], ignore_index=True) if history else None)
    frost = _frost_scores(weather, history)
    results: list[dict[str, Any]] = []
    for _, row in decades.iterrows():
        drought = _drought_component(row)
        sukhovei = float(np.clip(float(row.get("sukhovei_fraction", 0.0)), 0, 1))
        # Frost is attached to the decade containing the observed first frost.
        label = f"{int(row['month']):02d}-D{int(row['decade'])}"
        frost_component = 0.0
        if pd.notna(frost["first_frost_doy"]):
            first_date = pd.Timestamp(year=target_year, month=1, day=1) + pd.Timedelta(days=int(frost["first_frost_doy"]) - 1)
            if first_date.month == int(row["month"]) and ((first_date.day <= 10 and int(row["decade"]) == 1) or
                                                          (11 <= first_date.day <= 20 and int(row["decade"]) == 2) or
                                                          (first_date.day >= 21 and int(row["decade"]) == 3)):
                frost_component = frost["frost"]
        score = float(np.clip(max(drought, sukhovei, frost_component), 0, 1))
        notes = []
        if drought >= 0.6: notes.append("дефицит влаги")
        if sukhovei >= 0.4: notes.append("суховейные дни")
        if frost_component >= 0.4: notes.append("ранний заморозок")
        note = ", ".join(notes) if notes else "существенных сигналов риска нет"
        results.append({
            "decade_label": label, "risk_score": round(score, 3),
            "drought": round(drought, 3), "sukhovei": round(sukhovei, 3),
            "frost": round(frost_component, 3), "note": note,
        })
    return {"decades": results, "alert": any(x["risk_score"] >= 0.6 for x in results)}