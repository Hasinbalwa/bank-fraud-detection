
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pathlib import Path
import os

app = Flask(__name__)

# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_rf.pkl"

model = joblib.load(MODEL_PATH)

# Input columns used during model training
FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Please send transaction data as JSON"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing transaction features",
            "missing_features": missing
        }), 400

    try:
        transaction = pd.DataFrame(
            [[float(data[feature]) for feature in FEATURES]],
            columns=FEATURES
        )

        probability = model.predict_proba(transaction)[0][1]

        prediction = 1 if probability >= 0.40 else 0

        return jsonify({
            "prediction": prediction,
            "result": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": round(float(probability), 4)
        })

    except (ValueError, TypeError):
        return jsonify({"error": "All transaction features must be numeric"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
