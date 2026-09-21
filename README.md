# AgroAqkol — прогноз урожайности и риск засухи для Акмолинской области

## Быстрый старт на Windows (без `make`)

Открой PowerShell в этой папке и выполни по очереди, дожидаясь конца каждой команды:

```powershell
py -3.11 -m venv .venv

.venv\Scripts\activate

pip install -e ".[geo,dev]"

copy .env.example .env

python scripts/bootstrap_data.py --fields-csv data\fields.csv

python scripts/train_model.py

python -m uvicorn app.main:app --reload
```

Реальные данные (`data/raw/government/akmola_yield.xlsx`, `akmola_area.xlsx`) и демо-поля (`data/fields.csv`) уже лежат внутри этого архива — их скачивать и заполнять не нужно, можно сразу запускать.

После `uvicorn` открой в браузере:

`http://127.0.0.1:8000/docs`

Это Swagger, где можно проверить все эндпоинты:

1. `POST /fields` — создать поле, например:

   * `district: "Целиноградский"`
   * `crop: "Пшеница"`

   В ответ получишь `field_id`.

2. `GET /predict/{field_id}` — прогноз урожайности для поля.

3. `GET /risk/{field_id}` — риск по декадам.

4. `POST /chat` — AI-чат. Работает, если в `.env` указан `CHAT_LLM_API_KEY`.

---

## О проекте

**AgroAqkol** — система для хозяйств Акмолинской области.

По выбранному полю система предоставляет:

* прогноз урожайности;
* доверительный интервал прогноза;
* индекс риска засухи, суховея и раннего снега по декадам сезона;
* объяснение ключевых факторов, влияющих на прогноз;
* AI-чат для вопросов по конкретному полю.

Проект состоит из ML-пайплайна и backend-приложения на FastAPI.

## Структура

```text
.
├── src/crop_yield_prediction/  # ML: данные, обучение, прогноз, риск-индекс, реестр полей
├── app/                        # backend: FastAPI, AI-чат, роуты
├── scripts/                    # bootstrap_data.py, train_model.py
├── data/                       # сырые/обработанные данные
└── artifacts/                  # обученная модель и метрики
```

## Источники данных

1. **Урожайность по Акмолинской области** — реальные файлы уже проверены и подключены:

   `data/raw/government/akmola_yield.xlsx`

   Дополнительно:

   `data/raw/government/akmola_area.xlsx`

   Данные содержат информацию по районам и годам за 1991–2025 гг.

   В файле представлены блоки:

   * «Зерновые и бобовые»;
   * «из них: пшеницы».

   Урожайность указана в центнерах/га. Парсер автоматически переводит значения в т/га.

2. **Погода** — Open-Meteo Archive API, без ключа.

3. **Погода и влажность почвы** — NASA POWER, без ключа.

4. **Почва** — SoilGrids REST API, без ключа.

5. **NDVI** — Sentinel-2 через Microsoft Planetary Computer STAC, без ключа.

## Установка и запуск

### Windows

```powershell
py -3.11 -m venv .venv

.venv\Scripts\activate

pip install -e ".[geo,dev]"

copy .env.example .env
```

После этого:

```powershell
python scripts/bootstrap_data.py --fields-csv data\fields.csv

python scripts/train_model.py

python -m uvicorn app.main:app --reload
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[geo,dev]"

cp .env.example .env
```

Затем:

```bash
python scripts/bootstrap_data.py --fields-csv data/fields.csv

python scripts/train_model.py

python -m uvicorn app.main:app --reload
```

## API

* `POST /fields` — регистрация поля → `{ field_id }`
* `GET /predict/{field_id}` — прогноз урожайности (т/га) + доверительный интервал + топ-3 фактора (SHAP) + текстовое объяснение
* `GET /risk/{field_id}` — риск по декадам (0.0–1.0: засуха/суховей/заморозок) + `alert`
* `POST /chat` — AI-чат по конкретному полю, отвечает на основе его прогноза и риска

## Данные полей

Для работы с собственными полями используется:

```text
data/fields.csv
```

Район необходимо указывать так же, как он записан в исходных государственных данных.

Поддерживаемые районы:

* Аккольский
* Аршалынский
* Астраханский
* Атбасарский
* Биржан сал
* Буландынский
* Бурабайский
* Егиндыкольский
* Ерейментауский
* Есильский
* Жаксынский
* Жаркаинский
* Зерендинский
* Коргалжынский
* Сандыктауский
* Целиноградский
* Шортандинский

Также доступно значение `Всего по области` для общего показателя региона.

Поддерживаемые культуры:

* `Пшеница`
* `Зерновые и бобовые`

## Заметки

* Парсер государственных `.xlsx`-файлов является schema-tolerant, но использует узнаваемые русские/английские заголовки.
* NDVI требует установленного extra `geo`.
* Индекс риска — не отдельная ML-модель, а прозрачное правило поверх метеоданных.
* После обучения модели результаты и метрики сохраняются в папку `artifacts/`.
