from src.drift import detect_drift
from src.retrain import retrain

# Step 1: Check drift
drift = detect_drift("data/train.csv", "data/new_data.csv")
print("Drift scores:", drift)

# Step 2: Decide
threshold = 0.05

if any(p < threshold for p in drift.values()):
    print("Drift detected! Retraining model...")
    retrain()
else:
    print("No significant drift.")