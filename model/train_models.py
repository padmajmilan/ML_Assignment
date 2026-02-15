import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# Load dataset (use fixed CSV)
# -----------------------------
df = pd.read_csv("bank-full.csv", sep=",")

# -----------------------------
# FEATURE ENGINEERING (5+ CUSTOM FEATURES)
# -----------------------------

# 1. Age Group Binning - categorize age into meaningful groups
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 30, 45, 60, 100],
    labels=["Young", "Mid", "Senior", "Old"]
)

# 2. Contact Efficiency - ratio of duration to campaign calls
df["contact_efficiency"] = df["duration"] / (df["campaign"] + 1)

# 3. Previous Contact Indicator - binary flag for prior contact
df["had_previous_contact"] = np.where(df["previous"] > 0, 1, 0)

# 4. Balance Category - categorize balance into groups
df["balance_category"] = pd.cut(
    df["balance"],
    bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
    labels=["Negative", "Low", "Medium", "High", "Very_High"]
)

# 5. Log Transform Duration - reduce skewness
df["log_duration"] = np.log1p(df["duration"])

# 6. Days Since Last Contact Indicator
df["recently_contacted"] = np.where((df["pdays"] > 0) & (df["pdays"] < 30), 1, 0)

# 7. Job Risk Category - group jobs by risk/stability
high_risk_jobs = ["unemployed", "unknown", "student"]
medium_risk_jobs = ["self-employed", "entrepreneur", "services", "housemaid"]
df["job_risk"] = df["job"].apply(
    lambda x: "High" if x in high_risk_jobs else ("Medium" if x in medium_risk_jobs else "Low")
)

# 8. Total Contacts - sum of campaign and previous contacts
df["total_contacts"] = df["campaign"] + df["previous"]

# -----------------------------
# Encode categorical variables
# -----------------------------
# Store encoders for later use in the app
encoders = {}
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.extend(["age_group", "balance_category", "job_risk"])

for col in df.select_dtypes(include=["object", "category"]).columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    encoders[col] = encoder

# Save encoders
joblib.dump(encoders, "encoders.pkl")

# -----------------------------
# Train-test split
# -----------------------------
X = df.drop("y", axis=1)
y = df["y"]

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save test data for app
test_data = pd.DataFrame(X_test, columns=feature_names)
test_data["y"] = y_test.values
test_data.to_csv("test_data.csv", index=False)

# -----------------------------
# MODELS + HYPERPARAMETER TUNING
# -----------------------------

models = {}

# 1. Logistic Regression (Tuned)
log_reg_params = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs"],
    "class_weight": [None, "balanced"]
}
models["Logistic Regression"] = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    log_reg_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# 2. Decision Tree (Tuned)
dt_params = {
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "class_weight": [None, "balanced"]
}
models["Decision Tree"] = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# 3. KNN (Tuned)
knn_params = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
models["KNN"] = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# 4. Naive Bayes (with variance smoothing tuning)
nb_params = {
    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
}
models["Naive Bayes"] = GridSearchCV(
    GaussianNB(),
    nb_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# 5. Random Forest (Tuned)
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 10],
    "class_weight": [None, "balanced"]
}
models["Random Forest"] = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# 6. XGBoost (Tuned)
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0]
}
models["XGBoost"] = GridSearchCV(
    XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    ),
    xgb_params,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

# -----------------------------
# Train, Evaluate, Save
# -----------------------------
results = []
best_params_dict = {}

for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled data for distance-based models
    if name in ["KNN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    best_params = model.best_params_ if hasattr(model, "best_params_") else {}

    best_params_dict[name] = best_params

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    }

    results.append(metrics)
    joblib.dump(best_model, f"{name.replace(' ', '_')}.pkl")
    print(f"  Best params: {best_params}")
    print(f"  F1 Score: {metrics['F1']}, AUC: {metrics['AUC']}")

# Save metrics and best parameters
results_df = pd.DataFrame(results)
results_df.to_csv("model_metrics.csv", index=False)

# Save best parameters
params_df = pd.DataFrame([
    {"Model": k, "Best_Parameters": str(v)}
    for k, v in best_params_dict.items()
])
params_df.to_csv("best_params.csv", index=False)

print("\n" + "="*50)
print("✅ Training complete with feature engineering & tuning!")
print("="*50)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
