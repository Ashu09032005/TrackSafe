import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
import joblib
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

if not os.path.exists("labeled_train.csv"):
    raise FileNotFoundError("labeled_train.csv not found in current directory.")

df = pd.read_csv("labeled_train.csv")

X = df.drop(columns=["anomaly", "train_id", "timestamp", "latitude", "longitude"], errors='ignore')
y = df["anomaly"]

categorical = ["signal_status", "direction", "track_id", "weather_condition"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

label_map = {"none": 0, "signal_violation": 1, "track_conflict": 2, "overspeed": 3}
if y.dtype == 'object':
    y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

if issparse(X_train_enc):
    X_train_enc = X_train_enc.toarray()
    X_test_enc = X_test_enc.toarray()

X_res, y_res = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train_enc, y_train)

params = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
gs = GridSearchCV(GradientBoostingClassifier(), params, scoring='f1_weighted', cv=3)
gs.fit(X_res, y_res)

y_pred = gs.best_estimator_.predict(X_test_enc)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

joblib.dump(gs.best_estimator_, "best_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(label_map, "label_map.pkl")

model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_map = joblib.load("label_map.pkl")
inv_label_map = {v: k for k, v in label_map.items()}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = [
            "speed", "signal_distance", "train_length", "train_speed_limit",
            "distance_to_next_train", "brake_applied", "time_to_next_signal",
            "signal_visible", "signal_status", "direction", "track_id", "weather_condition"
        ]
        input_data = [data[field] for field in features]
        df_input = pd.DataFrame([dict(zip(features, input_data))])

        X_enc = preprocessor.transform(df_input)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()

        pred = model.predict(X_enc)[0]
        return jsonify({"predicted_anomaly": inv_label_map[pred]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

