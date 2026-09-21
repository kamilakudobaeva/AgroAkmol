"""Train and compare LinearRegression, RandomForest and XGBoost models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

TARGET = "yield_t_ha"
DROP = {TARGET}


def _prepare(df: pd.DataFrame):
    df = df.copy()
    # Keep year and location as numeric; district/crop are categorical.
    X = df.drop(columns=list(DROP), errors="ignore")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    # pandas 3.0 defaults string columns to a "str" dtype (neither classic "object"
    # nor the old "string[...]" extension dtype) — use the dtype-kind check so this
    # works across pandas 2.x and 3.x.
    cat = [c for c in X.columns if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return X, y, pre


def _metrics(name, model, X_test, y_test):
    pred = model.predict(X_test)
    return {"model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred)}


def main() -> None:
    data_path = ROOT / "data/processed/training_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Run bootstrap_data.py first: {data_path}")
    df = pd.read_csv(data_path)
    X, y, pre = _prepare(df)

    # Time-aware holdout: the newest year is test data to reduce leakage.
    years = pd.to_numeric(df.loc[y.index, "year"], errors="coerce")
    test_year = int(years.max())
    train_mask = years < test_year
    test_mask = years == test_year
    if train_mask.sum() < 5 or test_mask.sum() < 1:
        # Fallback for very small demo datasets.
        split = max(1, int(len(X) * 0.2))
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[-split:] = False
        test_mask = ~train_mask

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    models = {}
    lr = Pipeline([("prep", pre), ("model", LinearRegression())])
    models["LinearRegression"] = GridSearchCV(lr, {"model__fit_intercept": [True, False]}, cv=3, scoring="neg_mean_absolute_error")

    rf = Pipeline([("prep", pre), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
    models["RandomForest"] = GridSearchCV(rf, {
        "model__n_estimators": [200], "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 2],
    }, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)

    if XGBRegressor is not None:
        xgb = Pipeline([("prep", pre), ("model", XGBRegressor(objective="reg:squarederror", random_state=42,
                                                                n_estimators=300, tree_method="hist", n_jobs=4))])
        models["XGBoost"] = GridSearchCV(xgb, {
            "model__max_depth": [3, 5], "model__learning_rate": [0.03, 0.08],
            "model__subsample": [0.8], "model__colsample_bytree": [0.8],
        }, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)

    results = []
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        results.append(_metrics(name, model, X_test, y_test))

    metrics = pd.DataFrame(results).sort_values("MAE")
    metrics_dir = ROOT / "artifacts/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "model_comparison.csv", index=False)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name].best_estimator_

    # Bootstrap residuals give an honest, model-agnostic prediction interval without
    # depending on a specific XGBoost quantile objective/version.
    train_pred = best_model.predict(X_train)
    residuals = np.asarray(y_train) - np.asarray(train_pred)
    artifact = {"model": best_model, "residuals": residuals, "model_name": best_name}
    model_dir = ROOT / "artifacts/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / "best_model.joblib")

    # Save the raw model feature names after preprocessing for API metadata.
    feature_columns = list(X.columns)
    meta_dir = ROOT / "artifacts/metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "feature_columns.json").write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")
    (meta_dir / "model_metadata.json").write_text(json.dumps({"best_model": best_name, "test_year": test_year}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()
