import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

def retrain():
    train = pd.read_csv("data/train.csv")
    new = pd.read_csv("data/new_data.csv")

    combined = pd.concat([train, new])

    X = combined[["feature1", "feature2"]]
    y = combined["label"]

    model = LogisticRegression()
    model.fit(X, y)

    with open("models/model_v2.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model retrained and saved as v2.")