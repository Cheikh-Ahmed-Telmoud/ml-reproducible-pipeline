import yaml
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

df = pd.read_csv("data/processed/train.csv")
X = df.drop(columns=["target"])
y = df["target"]

model = Pipeline([
    ("scaler", StandardScaler()), 
    ("classifier", LogisticRegression(
        C=params["C"], max_iter=params["max_iter"], 
        random_state=params["random_state"]))
    ])

model.fit(X, y)

Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/model.joblib")

print("Entraînement terminé.")