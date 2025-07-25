# 🚆 TrackSafe: Real-Time Railway Anomaly Detection System

TrackSafe is a real-time machine learning solution designed to enhance railway safety by detecting potential **anomalies** using live sensor data. The system can flag conditions like:

-  **Signal Violations**
- **Overspeed**
- **Track Conflicts**

The trained model is deployed using a **Flask API**, which allows for real-time predictions via JSON input.

---

## Features

- **Multi-class Anomaly Detection** using a single model.
- **Real-Time Inference** via a REST API (Flask).
- **SMOTE** to handle imbalanced anomaly data.
- **Gradient Boosting Classifier** for high accuracy and robustness.

---

## Why Gradient Boosting?

Gradient Boosting is chosen because:

- It handles **non-linear relationships** and **feature interactions** very well.
- It is **robust to outliers**, which is helpful for noisy sensor data.
- It supports **multi-class classification** natively.
- It performs better than logistic regression or naive decision trees for this type of structured tabular data.

In our case, gradient boosting yielded better **F1 scores** and **precision on minority classes** (like 'signal_violation') when compared to random forests or logistic regression.

---

## Handling Imbalanced Data: SMOTE

Anomalies are **rare events**, making the dataset highly **imbalanced**.

To address this:
- We use **SMOTE (Synthetic Minority Over-sampling Technique)** to **generate synthetic samples** of minority classes during training.
- This helps the model learn the **true boundary** between normal and anomalous behavior.

Without SMOTE, the model tends to **ignore minority anomalies** like "track conflict".

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
```

<img width="1905" height="966" alt="image" src="https://github.com/user-attachments/assets/f147ee98-410e-4a47-9448-18db2825eac8" />
<img width="1918" height="917" alt="image" src="https://github.com/user-attachments/assets/aac2ab8b-f958-42de-974a-f64a3dea9c3d" />
<img width="1920" height="980" alt="image" src="https://github.com/user-attachments/assets/d41780f0-8d6c-4376-88b3-67d56b536699" />





