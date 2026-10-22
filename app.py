
from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]


@app.get("/")
def home():
    return jsonify({
        "project": "House Price Prediction API",
        "status": "running",
        "endpoints": ["/health", "/model-info", "/predict"]
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.get("/model-info")
def model_info():
    return jsonify({
        "features": FEATURES,
        "dataset": artifact["dataset"],
        "target_unit": artifact["target_unit"]
    })


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Send valid JSON."}), 400

    missing = [f for f in FEATURES if f not in payload]
    extra = [f for f in payload if f not in FEATURES]

    if missing or extra:
        return jsonify({
            "error": "Input features do not match.",
            "missing_features": missing,
            "unexpected_features": extra,
            "required_features": FEATURES
        }), 400

    try:
        values = {}

        for feature in FEATURES:
            value = payload[feature]

            if isinstance(value, bool):
                raise ValueError(f"{feature} must be numeric.")

            number = float(value)

            if not math.isfinite(number):
                raise ValueError(f"{feature} must be finite.")

            values[feature] = number

        input_data = pd.DataFrame([values], columns=FEATURES)
        prediction = float(model.predict(input_data)[0])
        price = prediction * 100000

        return jsonify({
            "estimated_price_usd": round(price, 2),
            "currency": "USD",
            "dataset": artifact["dataset"],
            "note": "Educational estimate, not a professional appraisal."
        })

    except (ValueError, TypeError, OverflowError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
