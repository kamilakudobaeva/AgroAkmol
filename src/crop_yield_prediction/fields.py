"""Field registry: the missing glue between POST /fields (mobile app) and
predict_yield / compute_risk, which take a FieldInput directly rather than a field_id.

In-memory + JSON-file persistence, good enough for a hackathon demo (single backend
process). Swap for a real DB later without changing the public function signatures.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from .data.akmola_sources import FieldInput

ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / "data" / "fields_store.json"

_lock = Lock()


def _load_store() -> dict[str, dict]:
    if not STORE_PATH.exists():
        return {}
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def _save_store(store: dict[str, dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def create_field(name: str, district: str, latitude: float, longitude: float,
                  crop: str, area_ha: float) -> str:
    """Register a new field (POST /fields body) and return its field_id."""
    field_id = uuid.uuid4().hex[:12]
    with _lock:
        store = _load_store()
        store[field_id] = {
            "name": name, "district": district, "latitude": latitude,
            "longitude": longitude, "crop": crop, "area_ha": area_ha,
        }
        _save_store(store)
    return field_id


def load_field(field_id: str) -> FieldInput:
    """Look up a registered field. Raises KeyError if unknown (→ 404 at the API layer)."""
    with _lock:
        store = _load_store()
    if field_id not in store:
        raise KeyError(field_id)
    rec = store[field_id]
    return FieldInput(
        district=rec["district"], latitude=rec["latitude"], longitude=rec["longitude"],
        crop=rec["crop"], area_ha=rec["area_ha"],
    )


def get_field_name(field_id: str) -> str:
    with _lock:
        store = _load_store()
    return store.get(field_id, {}).get("name", field_id)
