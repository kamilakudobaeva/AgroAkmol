"""Data sources and feature engineering for the AgroAqkol Akmola pipeline.

External sources used by this module:
- Kazakhstan Bureau of National Statistics XLSX export for district/crop yield.
- Open-Meteo Historical Weather API for daily weather.
- NASA POWER Daily API for an independent weather source and root-zone wetness.
- SoilGrids REST API for soil properties.
- Microsoft Planetary Computer STAC for Sentinel-2 NDVI.

The functions are intentionally schema-tolerant because the government XLSX layout
may change between exports.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

DEFAULT_START_MONTH = 5
DEFAULT_START_DAY = 1
DEFAULT_END_MONTH = 10
DEFAULT_END_DAY = 15


@dataclass(frozen=True)
class FieldInput:
    district: str
    latitude: float
    longitude: float
    crop: str
    area_ha: float


def _norm(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", "", text)


def _find_column(columns: Iterable[Any], groups: list[list[str]]) -> Any | None:
    """Return the first column whose normalized name contains all tokens of a group."""
    cols = list(columns)
    normalized = {c: _norm(c) for c in cols}
    for group in groups:
        for col in cols:
            name = normalized[col]
            if all(token in name for token in group):
                return col
    return None


def _read_excel_best_sheet(path: str | Path) -> pd.DataFrame:
    """Read the sheet with the most useful-looking tabular content."""
    path = Path(path)
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    best: pd.DataFrame | None = None
    best_score = -1
    for raw in sheets.values():
        if raw.empty:
            continue
        score = int(raw.notna().sum().sum())
        if score > best_score:
            best, best_score = raw.copy(), score
    if best is None:
        raise ValueError(f"No readable sheets in {path}")
    # Detect a header row by looking for likely Russian/English field labels.
    for i in range(min(15, len(best))):
        row = " ".join(_norm(v) for v in best.iloc[i].tolist())
        if any(x in row for x in ("район", "district", "урожайн", "yield", "культура", "crop")):
            best.columns = [str(v).strip() if not pd.isna(v) else f"unnamed_{j}" for j, v in enumerate(best.iloc[i])]
            return best.iloc[i + 1 :].reset_index(drop=True)
    best.columns = [str(v).strip() if not pd.isna(v) else f"unnamed_{j}" for j, v in enumerate(best.iloc[0])]
    return best.iloc[1:].reset_index(drop=True)


def parse_government_yield_xlsx(path: str | Path) -> pd.DataFrame:
    """Parse the Akmola government XLSX into the required long schema.

    Required output columns:
    year, district, crop, yield_t_ha, sown_area_ha, gross_output_t

    The parser supports both long tables and common year-as-columns exports.
    """
    raw = _read_excel_best_sheet(path)
    raw = raw.dropna(how="all").copy()

    year_col = _find_column(raw.columns, [["year"], ["год"]])
    district_col = _find_column(raw.columns, [["district"], ["район"], ["муницип"]])
    crop_col = _find_column(raw.columns, [["crop"], ["культур"], ["видсельхозкультур"]])
    yield_col = _find_column(raw.columns, [["yield"], ["урожайн"]])
    area_col = _find_column(raw.columns, [["sown", "area"], ["посев", "площад"], ["посевная", "площадь"]])
    gross_col = _find_column(raw.columns, [["gross", "output"], ["валов", "сбор"]])

    # Long format.
    if year_col and district_col and crop_col and (yield_col or area_col or gross_col):
        out = pd.DataFrame({
            "year": pd.to_numeric(raw[year_col], errors="coerce"),
            "district": raw[district_col].astype(str).str.strip(),
            "crop": raw[crop_col].astype(str).str.strip(),
            "yield_t_ha": pd.to_numeric(raw[yield_col], errors="coerce") if yield_col else np.nan,
            "sown_area_ha": pd.to_numeric(raw[area_col], errors="coerce") if area_col else np.nan,
            "gross_output_t": pd.to_numeric(raw[gross_col], errors="coerce") if gross_col else np.nan,
        })
    else:
        # Wide format: preserve descriptor columns and melt numeric year columns.
        year_columns: dict[Any, int] = {}
        for col in raw.columns:
            m = re.search(r"\b(19\d{2}|20\d{2})\b", str(col))
            if m:
                year_columns[col] = int(m.group(1))
        if not year_columns:
            raise ValueError("Could not detect a year column or year headers in government XLSX")

        district_col = district_col or _find_column(raw.columns, [["район"], ["district"]])
        crop_col = crop_col or _find_column(raw.columns, [["культур"], ["crop"]])
        if not district_col or not crop_col:
            raise ValueError("Could not detect district/crop columns in government XLSX")

        metric_col = _find_column(raw.columns, [["показател"], ["indicator"], ["наименован"]])
        if metric_col:
            records: list[dict[str, Any]] = []
            for _, row in raw.iterrows():
                metric = _norm(row[metric_col])
                for col, year in year_columns.items():
                    val = pd.to_numeric(row[col], errors="coerce")
                    if pd.isna(val):
                        continue
                    rec = {"year": year, "district": str(row[district_col]).strip(), "crop": str(row[crop_col]).strip(),
                           "yield_t_ha": np.nan, "sown_area_ha": np.nan, "gross_output_t": np.nan}
                    if "урожайн" in metric or "yield" in metric:
                        rec["yield_t_ha"] = val
                    elif "посев" in metric or "sown" in metric:
                        rec["sown_area_ha"] = val
                    elif "валов" in metric or "gross" in metric:
                        rec["gross_output_t"] = val
                    records.append(rec)
            out = pd.DataFrame(records)
            out = out.groupby(["year", "district", "crop"], as_index=False).first()
        else:
            raise ValueError("Wide XLSX needs an indicator/metric column to distinguish yield, area and gross output")

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out[out["year"].between(1950, datetime.now().year + 1, inclusive="both")]
    out["district"] = out["district"].replace({"nan": np.nan}).astype("string").str.strip()
    out["crop"] = out["crop"].replace({"nan": np.nan}).astype("string").str.strip()
    for col in ["yield_t_ha", "sown_area_ha", "gross_output_t"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["year", "district", "crop"]).copy()
    out["year"] = out["year"].astype(int)

    # If yield is absent but gross output and sown area are present, calculate it.
    missing_yield = out["yield_t_ha"].isna() & out["gross_output_t"].notna() & (out["sown_area_ha"] > 0)
    out.loc[missing_yield, "yield_t_ha"] = out.loc[missing_yield, "gross_output_t"] / out.loc[missing_yield, "sown_area_ha"]
    return out[["year", "district", "crop", "yield_t_ha", "sown_area_ha", "gross_output_t"]].reset_index(drop=True)


_CITY_ADMIN_DISTRICTS = {"г.а Косшы", "г.а. Кокшетау", "г.а. Степногорск"}


def _find_year_header_row(raw: pd.DataFrame) -> int:
    """Locate the row containing year numbers (1900-2100) in a wide district x year sheet."""
    for i in range(min(10, len(raw))):
        numeric = pd.to_numeric(raw.iloc[i], errors="coerce")
        if numeric.between(1900, 2100).sum() >= 5:
            return i
    raise ValueError("Could not locate a year header row in the government XLSX")


def parse_akmola_wide_district_year_xlsx(path: str | Path, sheet_name: int | str = 0,
                                          value_scale: dict[int, float] | None = None,
                                          default_crop: str = "Зерновые и бобовые") -> pd.DataFrame:
    """Parse the real Bureau of National Statistics wide-format export for Akmola region:
    districts as rows, years as columns, with one or more crop blocks per sheet separated
    by an "из них: <crop>" sub-header (e.g. the overall "Зерновые и бобовые" total, then a
    "из них: пшеницы" breakout). Handles the region-total row split across two cells
    ("Всего" / "по области").

    `value_scale`: optional {year: multiplier} to correct a unit switch mid-series
    (see `detect_area_unit_scale` — the sown-area export switches from thousand-ha to ha).

    Returns long format: year, district, crop, value.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    year_row = _find_year_header_row(raw)
    years = pd.to_numeric(raw.iloc[year_row], errors="coerce")

    records: list[dict[str, Any]] = []
    current_crop = default_crop
    pending_total = False

    for i in range(year_row + 1, len(raw)):
        label = raw.iloc[i, 0]
        if isinstance(label, str):
            clean = label.strip()
            m = re.match(r"из них:?\s*\n?(.+)", clean, re.IGNORECASE)
            if m:
                sub = m.group(1).strip().rstrip(":").lower()
                current_crop = "Пшеница" if sub.startswith("пшениц") else sub.capitalize()
                pending_total = False
                continue
            if clean.lower() == "всего":
                pending_total = True
                continue
            if pending_total and "област" in clean.lower():
                district, pending_total = "Всего по области", False
            elif clean:
                district, pending_total = clean, False
            else:
                continue
        else:
            continue  # blank/NaN label row: skip (region-total handled above, no other numeric-only rows expected)

        for j in range(1, raw.shape[1]):
            year = years.iloc[j]
            if pd.isna(year):
                continue
            val = pd.to_numeric(raw.iloc[i, j], errors="coerce")
            if pd.isna(val):
                continue
            scale = (value_scale or {}).get(int(year), 1.0)
            records.append({"year": int(year), "district": district, "crop": current_crop, "value": float(val) * scale})

    return pd.DataFrame.from_records(records)


def detect_area_unit_scale(path: str | Path, sheet_name: int | str = 0) -> dict[int, float]:
    """The sown-area export has a real quirk: years up to 1998 are reported in thousand
    hectares ('тыс.га'), 1999 onward in plain hectares ('в гектарах') - same sheet, same
    columns. Detects the switch column from those two marker strings and returns a
    per-year scale multiplier (1000.0 before the switch, 1.0 after)."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    year_row = _find_year_header_row(raw)
    years = pd.to_numeric(raw.iloc[year_row], errors="coerce")
    marker_row = raw.iloc[year_row - 1]
    switch_col = next((j for j, v in enumerate(marker_row) if isinstance(v, str) and "тыс" in v.lower()), None)
    scale: dict[int, float] = {}
    for j in range(1, raw.shape[1]):
        y = years.iloc[j]
        if pd.isna(y):
            continue
        scale[int(y)] = 1000.0 if (switch_col is not None and j <= switch_col) else 1.0
    return scale


def build_akmola_government_csv(yield_xlsx: str | Path, area_xlsx: str | Path | None,
                                 out_csv: str | Path, drop_city_admin: bool = True) -> pd.DataFrame:
    """End-to-end: parse the real stat.gov.kz yield + (optional) sown-area exports and
    write the long-format CSV (year, district, crop, yield_t_ha, sown_area_ha,
    gross_output_t) that build_training_dataset expects. Yield is reported in
    centners/ha (ц/га) - converted to tonnes/ha (÷10) to match the project's yield_t_ha
    convention everywhere else."""
    yield_long = parse_akmola_wide_district_year_xlsx(yield_xlsx)
    yield_long["yield_t_ha"] = yield_long["value"] / 10.0

    if area_xlsx is not None:
        area_scale = detect_area_unit_scale(area_xlsx)
        area_long = parse_akmola_wide_district_year_xlsx(area_xlsx, value_scale=area_scale)
        area_long = area_long.rename(columns={"value": "sown_area_ha"})
        merged = yield_long.merge(area_long[["year", "district", "crop", "sown_area_ha"]],
                                   on=["year", "district", "crop"], how="left")
        merged["gross_output_t"] = merged["yield_t_ha"] * merged["sown_area_ha"]
    else:
        merged = yield_long.copy()
        merged["sown_area_ha"] = np.nan
        merged["gross_output_t"] = np.nan

    if drop_city_admin:
        merged = merged[~merged["district"].isin(_CITY_ADMIN_DISTRICTS)]

    out = merged[["year", "district", "crop", "yield_t_ha", "sown_area_ha", "gross_output_t"]] \
        .sort_values(["district", "crop", "year"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def fetch_open_meteo_daily(latitude: float, longitude: float, start: str, end: str,
                           timeout: int = 60) -> pd.DataFrame:
    """Download daily historical weather from Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "daily": ",".join([
            "temperature_2m_min", "temperature_2m_mean", "temperature_2m_max",
            "precipitation_sum", "relative_humidity_2m_mean", "wind_speed_10m_max",
            "wind_speed_10m_mean", "snowfall_sum",
        ]),
        "timezone": "Asia/Almaty",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "daily" not in data:
        raise ValueError(f"Open-Meteo response has no daily data: {data}")
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df.pop("time"))
    return df


def fetch_nasa_power_daily(latitude: float, longitude: float, start: str, end: str,
                           timeout: int = 60) -> pd.DataFrame:
    """Download daily NASA POWER Agroclimatology variables, including GWETROOT."""
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR,GWETROOT",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "format": "JSON",
    }
    r = requests.get(NASA_POWER_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    values = data.get("properties", {}).get("parameter", {})
    if not values:
        raise ValueError(f"NASA POWER response has no parameter data: {data}")
    df = pd.DataFrame(values)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.index.name = "date"
    df = df.reset_index()
    return df.rename(columns={"T2M": "nasa_t2m", "T2M_MAX": "nasa_t2m_max", "T2M_MIN": "nasa_t2m_min",
                              "RH2M": "nasa_rh", "WS10M": "nasa_wind", "PRECTOTCORR": "nasa_precip",
                              "GWETROOT": "gwetroot"})


def aggregate_decades(weather: pd.DataFrame, nasa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate daily weather into calendar decades and add dry-wind/GDD features."""
    df = weather.copy()
    df["date"] = pd.to_datetime(df["date"])
    if nasa is not None and not nasa.empty:
        n = nasa.copy()
        n["date"] = pd.to_datetime(n["date"])
        df = df.merge(n, on="date", how="left")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["decade"] = np.where(df["day"] <= 10, 1, np.where(df["day"] <= 20, 2, 3))
    df["decade_label"] = df["date"].dt.strftime("%Y-%m") + "-D" + df["decade"].astype(str)
    df["sukhovei_day"] = (
        (df["temperature_2m_max"] > 25)
        & (df.get("relative_humidity_2m_mean", pd.Series(index=df.index, dtype=float)) < 30)
        & (df["wind_speed_10m_max"] > 5)
    ).astype(int)
    df["gdd_base5"] = (df["temperature_2m_mean"] - 5).clip(lower=0)
    grouped = df.groupby(["year", "month", "decade"], as_index=False).agg(
        temp_min=("temperature_2m_min", "mean"),
        temp_mean=("temperature_2m_mean", "mean"),
        temp_max=("temperature_2m_max", "mean"),
        precipitation_mm=("precipitation_sum", "sum"),
        humidity_mean=("relative_humidity_2m_mean", "mean"),
        wind_mean_ms=("wind_speed_10m_mean", "mean"),
        wind_max_ms=("wind_speed_10m_max", "max"),
        sukhovei_fraction=("sukhovei_day", "mean"),
        gdd_base5=("gdd_base5", "sum"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    if "gwetroot" in df:
        gw = df.groupby(["year", "month", "decade"], as_index=False)["gwetroot"].mean()
        grouped = grouped.merge(gw, on=["year", "month", "decade"], how="left")
    return grouped


def precipitation_anomaly(decades: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add decade precipitation anomaly as percent of historical mean and z-score."""
    df = decades.copy()
    keys = ["month", "decade"]
    ref = history if history is not None else df
    stats = ref.groupby(keys)["precipitation_mm"].agg(["mean", "std"]).reset_index()
    stats = stats.rename(columns={"mean": "precip_norm_mm", "std": "precip_std_mm"})
    df = df.merge(stats, on=keys, how="left")
    df["precip_anomaly_pct"] = np.where(df["precip_norm_mm"] > 0,
                                         (df["precipitation_mm"] / df["precip_norm_mm"] - 1) * 100, 0)
    df["precip_anomaly_z"] = np.where(df["precip_std_mm"] > 0,
                                       (df["precipitation_mm"] - df["precip_norm_mm"]) / df["precip_std_mm"], 0)
    return df


def fetch_soilgrids(latitude: float, longitude: float, properties: tuple[str, ...] = ("clay", "sand", "silt", "soc", "phh2o")) -> dict[str, float]:
    """Query SoilGrids for a point and return topsoil means where available."""
    params = {"lon": longitude, "lat": latitude, "property": list(properties), "depth": ["0-5cm"], "value": "mean"}
    r = requests.get(SOILGRIDS_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    result: dict[str, float] = {}
    for layer in data.get("properties", {}).get("layers", []):
        name = layer.get("name")
        vals = layer.get("depths", [])
        if not name or not vals:
            continue
        value = vals[0].get("values", {}).get("mean")
        if value is not None:
            result[f"soil_{name}_0_5cm"] = float(value)
    return result


def fetch_sentinel2_ndvi(latitude: float, longitude: float, start: str, end: str,
                         max_cloud: float = 40.0, max_items: int = 6) -> pd.DataFrame:
    """Get Sentinel-2 L2A NDVI near a field from Microsoft Planetary Computer STAC.

    Requires optional dependencies: pystac-client, planetary-computer, stackstac,
    xarray, rasterio, shapely.
    """
    try:
        import planetary_computer
        import pystac_client
        import stackstac
        from shapely.geometry import Point
    except ImportError as exc:
        raise RuntimeError("Install optional NDVI dependencies: pystac-client planetary-computer stackstac shapely rasterio xarray") from exc

    catalog = pystac_client.Client.open(PLANETARY_COMPUTER_STAC)
    point = Point(float(longitude), float(latitude))
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=point.__geo_interface__,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=max_items,
    )
    items = list(search.items())
    if not items:
        return pd.DataFrame(columns=["date", "ndvi"])

    signed = [planetary_computer.sign(item) for item in items]
    arr = stackstac.stack(signed, assets=["B04", "B08"], bounds_latlon=(longitude - 0.001, latitude - 0.001, longitude + 0.001, latitude + 0.001), epsg=4326, resolution=0.0001)
    # Average over the tiny point window, then compute NDVI.
    red = arr.sel(band="B04").mean(dim=[d for d in arr.sel(band="B04").dims if d in ("x", "y")], skipna=True)
    nir = arr.sel(band="B08").mean(dim=[d for d in arr.sel(band="B08").dims if d in ("x", "y")], skipna=True)
    ndvi = ((nir - red) / (nir + red)).compute()
    dates = pd.to_datetime(ndvi["time"].values).tz_localize(None)
    return pd.DataFrame({"date": dates, "ndvi": np.asarray(ndvi.values, dtype=float)})


def make_decade_features(decades: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    """Flatten decade records into a model-ready feature dictionary."""
    result: dict[str, float] = {}
    for _, row in decades.iterrows():
        label = f"{int(row['month']):02d}_d{int(row['decade'])}"
        for col in ["temp_min", "temp_mean", "temp_max", "precipitation_mm", "humidity_mean",
                    "wind_mean_ms", "wind_max_ms", "sukhovei_fraction", "gdd_base5",
                    "gwetroot", "precip_anomaly_pct", "precip_anomaly_z"]:
            if col in row.index and pd.notna(row[col]):
                result[f"{prefix}weather_{label}_{col}"] = float(row[col])
    return result


def build_training_dataset(fields_csv: str | Path, government_csv: str | Path,
                           output_csv: str | Path, start_month: int = DEFAULT_START_MONTH,
                           end_month: int = DEFAULT_END_MONTH, include_ndvi: bool = False,
                           lag_years: int = 3) -> pd.DataFrame:
    """Build one row per field/crop/year by joining yield, weather, soil and optional NDVI."""
    fields = pd.read_csv(fields_csv)
    required = {"district", "latitude", "longitude", "crop", "area_ha"}
    missing = required - set(fields.columns)
    if missing:
        raise ValueError(f"fields CSV is missing columns: {sorted(missing)}")
    gov = pd.read_csv(government_csv)
    years = sorted(int(y) for y in gov["year"].dropna().unique())
    records: list[dict[str, Any]] = []
    soil_cache: dict[tuple[float, float], dict[str, float]] = {}

    for _, field in fields.iterrows():
        district = str(field["district"]).strip()
        crop = str(field["crop"]).strip()
        lat, lon = float(field["latitude"]), float(field["longitude"])
        for year in years:
            ymatch = gov[(gov["year"] == year) & (gov["district"].astype(str).str.strip() == district) & (gov["crop"].astype(str).str.strip() == crop)]
            if ymatch.empty:
                continue
            start = date(year, start_month, 1)
            end = date(year, end_month, 15)
            print(f"Weather: {district}/{crop}/{year}")
            weather = fetch_open_meteo_daily(lat, lon, start.isoformat(), end.isoformat())
            nasa = fetch_nasa_power_daily(lat, lon, start.isoformat(), end.isoformat())
            decades = precipitation_anomaly(aggregate_decades(weather, nasa))
            feats: dict[str, Any] = {
                "year": year, "district": district, "crop": crop,
                "latitude": lat, "longitude": lon, "area_ha": float(field["area_ha"]),
                "yield_t_ha": float(ymatch.iloc[0]["yield_t_ha"]) if pd.notna(ymatch.iloc[0]["yield_t_ha"]) else np.nan,
            }
            feats.update(make_decade_features(decades))
            key = (round(lat, 5), round(lon, 5))
            if key not in soil_cache:
                try:
                    soil_cache[key] = fetch_soilgrids(lat, lon)
                except Exception as exc:
                    print(f"SoilGrids warning for {lat},{lon}: {exc}")
                    soil_cache[key] = {}
                time.sleep(0.25)
            feats.update(soil_cache[key])
            if include_ndvi:
                try:
                    ndvi = fetch_sentinel2_ndvi(lat, lon, start.isoformat(), end.isoformat())
                    if not ndvi.empty:
                        feats["ndvi_mean"] = float(ndvi["ndvi"].mean())
                        feats["ndvi_max"] = float(ndvi["ndvi"].max())
                        feats["ndvi_last"] = float(ndvi.sort_values("date").iloc[-1]["ndvi"])
                except Exception as exc:
                    print(f"NDVI warning for {lat},{lon},{year}: {exc}")
            records.append(feats)

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No training rows were created. Check district/crop names and fields CSV.")
    # Lag features are computed only from historical government yield, never from the target row itself.
    gov_key = gov.copy()
    gov_key["district"] = gov_key["district"].astype(str).str.strip()
    gov_key["crop"] = gov_key["crop"].astype(str).str.strip()
    for lag in range(1, lag_years + 1):
        lookup = gov_key[["year", "district", "crop", "yield_t_ha"]].copy()
        lookup["year"] += lag
        lookup = lookup.rename(columns={"yield_t_ha": f"yield_lag_{lag}"})
        out = out.merge(lookup, on=["year", "district", "crop"], how="left")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
