# TrackSafe

This project trains a machine learning model to detect potential railway anomalies in real time — such as signal violations, overspeed, and track conflicts — using live input data. The model is served via a Flask API endpoint and supports JSON input for real-time predictions.

---

## Project Structure

- `Realtime Anomaly App.py` – Unified script for:
  - Data loading and preprocessing
  - Model training using GridSearchCV and SMOTE
  - Model serialization
  - Flask server deployment for real-time predictions
- `labeled_train.csv` – CSV file containing labeled historical data used to train the model
- `best_model.pkl` – Serialized trained ML model
- `preprocessor.pkl` – Saved OneHotEncoder and ColumnTransformer
- `label_map.pkl` – Mapping of anomaly labels to numeric classes

---

## Model Information

- **Algorithm**: Gradient Boosting Classifier (from `sklearn`)
- **Hyperparameter Tuning**: `GridSearchCV` with parameters:
  - `n_estimators`: 100, 200
  - `max_depth`: 5, 10
- **Preprocessing**:
  - `OneHotEncoder` for categorical features (`signal_status`, `direction`, `track_id`, `weather_condition`)
  - `ColumnTransformer` to combine categorical and numerical data
- **Imbalanced Data Handling**: `SMOTE` (Synthetic Minority Over-sampling Technique)
- **Model Evaluation**: `accuracy_score` and `f1_score` (weighted)

---

## Installation & Setup

### 1. Clone the repository or download files

### 2. Install Python dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn flask joblib

<img width="1905" height="966" alt="image" src="https://github.com/user-attachments/assets/f147ee98-410e-4a47-9448-18db2825eac8" />
<img width="1918" height="917" alt="image" src="https://github.com/user-attachments/assets/aac2ab8b-f958-42de-974a-f64a3dea9c3d" />
<img width="1920" height="980" alt="image" src="https://github.com/user-attachments/assets/d41780f0-8d6c-4376-88b3-67d56b536699" />


