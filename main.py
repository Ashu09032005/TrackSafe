# app.py (Flask Backend)
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Required fields (must match training feature order)
        features = [
            "speed",
            "signal_distance",
            "train_length",
            "train_speed_limit",
            "distance_to_next_train",
            "brake_applied",
            "time_to_next_signal",
            "signal_visible",
            "signal_status",
            "direction",
            "track_id",
            "weather_condition"
        ]

        input_data = [data[field] for field in features]
        input_df = [dict(zip(features, input_data))]

        import pandas as pd
        df = pd.DataFrame(input_df)

        X_processed = preprocessor.transform(df)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        prediction = model.predict(X_processed)[0]

        label_map_reverse = {
            0: "none",
            1: "signal_violation",
            2: "track_conflict",
            3: "overspeed"
        }

        return jsonify({"predicted_anomaly": label_map_reverse[prediction]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
