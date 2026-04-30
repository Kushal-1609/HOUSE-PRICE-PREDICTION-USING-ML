from __future__ import annotations

import pickle
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PROJECT_ROOT = Path(__file__).resolve().parent
COMBINED_MODEL_PATH = PROJECT_ROOT / "combined_model.pkl"

MODEL: Any | None = None
MODEL_LOAD_ERROR: str | None = None
MODEL_LOCK = Lock()


class PredictionError(Exception):
    pass


def load_model_once() -> Any:
    """Load the combined_model.pkl once (thread-safe)."""
    global MODEL, MODEL_LOAD_ERROR

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        if MODEL_LOAD_ERROR is not None:
            raise RuntimeError(MODEL_LOAD_ERROR)

        if not COMBINED_MODEL_PATH.exists():
            MODEL_LOAD_ERROR = f"{COMBINED_MODEL_PATH.name} not found"
            raise RuntimeError(MODEL_LOAD_ERROR)

        try:
            with COMBINED_MODEL_PATH.open("rb") as f:
                artifact = pickle.load(f)
        except Exception as exc:
            MODEL_LOAD_ERROR = f"Failed to load {COMBINED_MODEL_PATH.name}: {exc}"
            raise RuntimeError(MODEL_LOAD_ERROR) from exc

        # artifact can be pipeline or dict with "model"
        model = artifact.get("model") if isinstance(artifact, dict) else artifact

        if not hasattr(model, "predict"):
            MODEL_LOAD_ERROR = "Loaded artifact does not expose predict()"
            raise RuntimeError(MODEL_LOAD_ERROR)

        MODEL = model
        print(f"[MODEL] Loaded combined artifact from {COMBINED_MODEL_PATH}")
        return MODEL


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Convert JSON → DataFrame
        df = pd.DataFrame([payload])

        model = load_model_once()

        # Debug (optional)
        print(f"[DEBUG] incoming_columns = {list(df.columns)}")
        print(f"[DEBUG] model_feature_names_in_ = {getattr(model, 'feature_names_in_', None)}")

        # Predict (log scale)
        preds = model.predict(df)

        # Extract scalar
        if hasattr(preds, "ravel"):
            raw_value = float(preds.ravel()[0])
        else:
            raw_value = float(preds[0])

        # 🔥 CRITICAL FIX → reverse log transform
        price = float(np.exp(raw_value))

        return jsonify({"predicted_price": price})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/")
def home():
    return "API running"


if __name__ == "__main__":
    try:
        load_model_once()
    except Exception as exc:
        print(f"Startup error: {exc}")

    app.run(host="0.0.0.0", port=10000)