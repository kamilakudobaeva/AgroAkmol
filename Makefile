PYTHON ?= python3.11

.PHONY: install bootstrap train api test mobile

install:
	$(PYTHON) -m pip install -e ".[geo,dev]"

bootstrap:
	$(PYTHON) scripts/bootstrap_data.py --fields-csv data/fields.csv

train:
	$(PYTHON) scripts/train_model.py

api:
	$(PYTHON) -m uvicorn app.main:app --reload

test:
	pytest

mobile:
	cd mobile && flutter pub get && flutter run
