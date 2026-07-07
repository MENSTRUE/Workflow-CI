import os
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "preprocessing" / "nearest_earth_object_preprocessing"

X_train_path = DATA_DIR / "X_train.csv"
y_train_path = DATA_DIR / "y_train.csv"
X_test_path = DATA_DIR / "X_test.csv"
y_test_path = DATA_DIR / "y_test.csv"


mlflow.set_tracking_uri(
    "https://dagshub.com/MENSTRUE/Eksperimen_SML_wafa_bila_syaefurokhman.mlflow"
)
mlflow.set_experiment("NASA_Asteroid_Workflow_CI")

# Penting untuk GitHub Actions saat pakai `mlflow run`.
# `mlflow run` otomatis membuat MLFLOW_RUN_ID lokal.
# Kalau tracking URI diarahkan ke DagsHub, run lokal itu tidak ditemukan di DagsHub.
# Jadi env run lama harus dibersihkan sebelum start_run baru.
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)


X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()

if X_test_path.exists() and y_test_path.exists():
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Pastikan nama kolom X_test sama dengan X_train.
    # Beberapa file preprocessing lama menyimpan X_test dengan kolom 0,1,2,3,4.
    if list(X_test.columns) != list(X_train.columns):
        if X_test.shape[1] == X_train.shape[1]:
            X_test.columns = X_train.columns
        else:
            raise ValueError(
                f"Jumlah fitur X_test ({X_test.shape[1]}) tidak sama "
                f"dengan X_train ({X_train.shape[1]})."
            )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )


params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
}

with mlflow.start_run(run_name="workflow_ci_random_forest") as run:
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    conda_env = {
        "name": "nasa_model_env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "cloudpickle",
                    "pandas",
                    "numpy",
                    "scikit-learn",
                ]
            },
        ],
    }

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_test.head(3),
        conda_env=conda_env,
    )

    run_id = run.info.run_id
    run_id_file = REPO_DIR / "run_id.txt"
    run_id_file.write_text(run_id)

    print("Modelling CI Selesai!")
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    conda_env = {
        "name": "nasa_model_env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "cloudpickle",
                    "pandas",
                    "numpy",
                    "scikit-learn",
                ]
            },
        ],
    }

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_test.head(3),
        conda_env=conda_env,
    )

    run_id = run.info.run_id
    run_id_file = REPO_DIR / "run_id.txt"
    run_id_file.write_text(run_id)

    print("Modelling CI Selesai!")
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
