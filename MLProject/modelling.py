import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Konfigurasi Kredensial (Bypass Login Manual)
OWNER = 'MENSTRUE'
REPO_NAME = 'Eksperimen_SML_wafa_bila_syaefurokhman'

# Mengambil token dari GitHub Secrets yang sudah kamu pasang
token = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Setup URI dan Environment secara manual agar GitHub Actions bisa login
mlflow.set_tracking_uri(f"https://dagshub.com/{OWNER}/{REPO_NAME}.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = OWNER
os.environ['MLFLOW_TRACKING_PASSWORD'] = token

# 2. Path Data
X_path = 'preprocessing/nearest_earth_object_preprocessing/X_train.csv'
y_path = 'preprocessing/nearest_earth_object_preprocessing/y_train.csv'

X_train = pd.read_csv(X_path)
y_train = pd.read_csv(y_path)

# 3. Autolog dan Training
mlflow.autolog()

with mlflow.start_run(run_name="NASA_Asteroid_CI_Retraining"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train.values.ravel())

    print("Modelling CI Selesai! Cek DagsHub sekarang.")