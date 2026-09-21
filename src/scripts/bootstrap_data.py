"""Download/prepare Akmola data for AgroAqkol.

Real stat.gov.kz exports needed (district x year wide format, see README):
  - "Урожайность зерновых (включая рис) и бобовых культур" (yield, ц/га)
  - "Площадь зерновых культур" (sown area) - optional, only used for gross_output_t

Example:
  python scripts/bootstrap_data.py --fields-csv data/fields.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from crop_yield_prediction.data.akmola_sources import build_akmola_government_csv, build_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gov-yield-xlsx", default="data/raw/government/akmola_yield.xlsx",
                         help="stat.gov.kz 'Урожайность ...' export (district x year, ц/га)")
    parser.add_argument("--gov-area-xlsx", default="data/raw/government/akmola_area.xlsx",
                         help="stat.gov.kz 'Площадь ...' export (district x year, sown area). Optional.")
    parser.add_argument("--fields-csv", default="data/fields.csv")
    parser.add_argument("--output", default="data/processed/training_dataset.csv")
    parser.add_argument("--with-ndvi", action="store_true")
    args = parser.parse_args()

    gov_yield_xlsx = ROOT / args.gov_yield_xlsx
    gov_area_xlsx = ROOT / args.gov_area_xlsx
    gov_csv = ROOT / "data/processed/akmola_yield.csv"

    if not gov_yield_xlsx.exists():
        raise FileNotFoundError(
            f"Government yield XLSX not found: {gov_yield_xlsx}. "
            "Download 'Урожайность зерновых (включая рис) и бобовых культур' from stat.gov.kz."
        )
    area_path = gov_area_xlsx if gov_area_xlsx.exists() else None
    if area_path is None:
        print(f"Note: {gov_area_xlsx} not found - proceeding without sown-area/gross_output_t (yield-only is fine).")

    gov = build_akmola_government_csv(gov_yield_xlsx, area_path, gov_csv)
    print(f"Saved {len(gov):,} government rows -> {gov_csv}")
    print(f"Districts: {sorted(gov['district'].unique())}")
    print(f"Crops: {sorted(gov['crop'].unique())}")

    fields = ROOT / args.fields_csv
    if not fields.exists():
        print(f"Government data is ready. Create {fields} with columns: district,latitude,longitude,crop,area_ha")
        return
    out = build_training_dataset(fields, gov_csv, ROOT / args.output, include_ndvi=args.with_ndvi)
    print(f"Saved {len(out):,} training rows -> {ROOT / args.output}")


if __name__ == "__main__":
    main()
