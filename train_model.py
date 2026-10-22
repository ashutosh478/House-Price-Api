
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

df.to_csv(DATA_DIR / "house_prices.csv", index=False)

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 15, 25],
    "model__min_samples_leaf": [1, 2]
}

search = GridSearchCV(
    pipeline,
    params,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("Training and tuning model...")
search.fit(X_train, y_train)

best_model = search.best_estimator_
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

artifact = {
    "model": best_model,
    "features": list(X.columns),
    "dataset": "California Housing",
    "target_unit": "100000 USD"
}

joblib.dump(artifact, MODEL_DIR / "house_price_model.joblib")

metrics = {
    "best_params": search.best_params_,
    "cv_mae": float(-search.best_score_),
    "test_mae": float(mae),
    "test_rmse": float(rmse),
    "test_r2": float(r2)
}

(MODEL_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=2)
)

print("\nTraining complete!")
print("Best parameters:", search.best_params_)
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2: {r2:.4f}")
print("Saved model in models/house_price_model.joblib")
