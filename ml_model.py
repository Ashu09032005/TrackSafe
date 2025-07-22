import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
import joblib

# Step 1: Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    "train_id": range(n_samples),
    "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
    "latitude": np.random.uniform(20.0, 25.0, n_samples),
    "longitude": np.random.uniform(75.0, 80.0, n_samples),
    "speed": np.random.normal(80, 15, n_samples).clip(0),
    "signal_distance": np.random.uniform(0, 5, n_samples),
    "train_length": np.random.randint(100, 2000, n_samples),
    "train_speed_limit": np.random.choice([60, 80, 100, 120], n_samples),
    "distance_to_next_train": np.random.uniform(0, 2, n_samples),
    "brake_applied": np.random.choice([0, 1], n_samples),
    "time_to_next_signal": np.random.randint(1, 1000, n_samples),  # Increased max to simulate distant signals
    "signal_visible": np.random.choice([0, 1], n_samples),
    "signal_status": np.random.choice(["RED", "GREEN", "YELLOW"], n_samples),
    "direction": np.random.choice(["N", "S", "E", "W"], n_samples),
    "track_id": np.random.choice(["A", "B", "C"], n_samples),
    "weather_condition": np.random.choice(["clear", "rain", "fog"], n_samples)
})

# Threshold for when to consider a signal violation (seconds)
SIGNAL_VIOLATION_TIME_THRESHOLD = 200

# Step 2: Rule-based anomaly generation
def generate_anomaly(row):
    if row['speed'] > row['train_speed_limit'] + 30:
        return "overspeed"
    elif (
        row['signal_status'] == "RED"
        and row['signal_distance'] < 0.5
        and row['brake_applied'] == 0
        and row['time_to_next_signal'] < SIGNAL_VIOLATION_TIME_THRESHOLD
    ):
        return "signal_violation"
    elif row['distance_to_next_train'] < 0.2 and row['speed'] > 60:
        return "track_conflict"
    else:
        return "none"

df["anomaly"] = df.apply(generate_anomaly, axis=1)
df.to_csv("train.csv", index=False)

# Step 3: Data Preparation
X = df.drop(columns=["anomaly", "train_id", "timestamp", "latitude", "longitude"])
y = df["anomaly"]

categorical = ["signal_status", "direction", "track_id", "weather_condition"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

label_map = {"none": 0, "signal_violation": 1, "track_conflict": 2, "overspeed": 3}
y = y.map(label_map)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

if issparse(X_train_enc):
    X_train_enc = X_train_enc.toarray()
    X_test_enc = X_test_enc.toarray()

# Step 5: SMOTE
sm = SMOTE(random_state=42, k_neighbors=3)
X_res, y_res = sm.fit_resample(X_train_enc, y_train)

# Step 6: Model Training
models = {
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "Adaboost": AdaBoostClassifier(),
    "Gradient Boost": GradientBoostingClassifier(),
    "XGB": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_res, y_res)
    y_pred_test = model.predict(X_test_enc)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("F1 Score:", f1_score(y_test, y_pred_test, average='weighted'))
    print("Classification Report:\n", classification_report(y_test, y_pred_test, zero_division=0))

# Step 7: Grid Search on GradientBoost
params = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
gs = GridSearchCV(GradientBoostingClassifier(), params, scoring='f1_weighted', cv=3, verbose=1)
gs.fit(X_res, y_res)

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test_enc)
print("\nBest GradientBoost Parameters:", gs.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Step 8: Save the trained model and preprocessor
joblib.dump(best_model, "best_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
