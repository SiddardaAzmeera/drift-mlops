from fastapi import FastAPI
from src.drift import detect_drift
from src.retrain import retrain

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Drift MLOps API is running"}

@app.get("/check-drift")
def check_drift():
    drift = detect_drift("data/train.csv", "data/new_data.csv")
    
    threshold = 0.05
    drift_detected = any(p < threshold for p in drift.values())

    return {
        "drift_scores": drift,
        "drift_detected": drift_detected
    }

@app.post("/retrain")
def retrain_model():
    retrain()
    return {"message": "Model retrained successfully"}