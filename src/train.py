import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data/train.csv")

X = df[["feature1", "feature2"]]
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

with open("models/model_v1.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")