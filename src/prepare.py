import yaml
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["prepare"]

dataset = load_breast_cancer(as_frame=True)
df = dataset.frame

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], 
    random_state=params["random_state"], 
    stratify=y
)

Path("data/processed").mkdir(parents=True, exist_ok=True)

train = X_train.copy()
train["target"] = y_train
test = X_test.copy()
test["target"] = y_test

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Préparation terminée.")