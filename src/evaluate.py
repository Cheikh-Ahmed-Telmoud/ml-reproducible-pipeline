import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test = pd.read_csv("data/processed/test.csv")
X_test = test.drop(columns=["target"])
y_test = test["target"]

model = joblib.load("models/model.joblib")
predictions = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions)),
    "recall": float(recall_score(y_test, predictions)),
    "f1_score": float(f1_score(y_test, predictions))
}

Path("reports").mkdir(exist_ok=True)

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)