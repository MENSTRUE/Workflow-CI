import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Setup tracking ke DagsHub (Username & Password diambil dari ENV GitHub Actions)
mlflow.set_tracking_uri("https://dagshub.com/MENSTRUE/Eksperimen_SML_wafa_bila_syaefurokhman.mlflow")

# PATH DATA: Harus relatif terhadap lokasi file MLProject
# Karena running di GitHub Actions, path-nya adalah:
X_path = 'MLProject/preprocessing/nearest_earth_object_preprocessing/X_train.csv'
y_path = 'MLProject/preprocessing/nearest_earth_object_preprocessing/y_train.csv'

X_train = pd.read_csv(X_path)
y_train = pd.read_csv(y_path)

# Aktifkan autolog agar tercatat otomatis ke Run yang dibuat oleh Workflow CI
mlflow.sklearn.autolog()

# LANGSUNG TRAINING (Tanpa start_run manual)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train.values.ravel())

print("Modelling CI Selesai!")