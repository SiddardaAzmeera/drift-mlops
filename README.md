# Data Drift Detection & Auto-Retraining Pipeline

## Overview
This project simulates a real-world MLOps pipeline that:
- Trains an ML model
- Detects data drift using statistical tests
- Automatically retrains the model when drift is detected

## Features
- Drift detection using Kolmogorov-Smirnov test
- Automated retraining pipeline
- Model versioning (v1 → v2)
- Modular code structure

## Tech Stack
- Python
- scikit-learn
- pandas
- scipy

## How It Works
1. Train initial model on baseline data
2. Compare new incoming data with training data
3. Detect drift using p-values
4. If drift detected → retrain model automatically

## Run Instructions
```bash
python src/train.py
python main.py

Output
model_v1.pkl (initial model)
model_v2.pkl (retrained model after drift)
Author

Siddarda Azmeera
Aspiring AI / MLOps Engineer