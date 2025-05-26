import pandas as pd
import os
import joblib
import json
import argparse

from libml.preprocessing import build_pipeline
from sklearn.metrics import accuracy_score

def train(input_path: str, model_path: str, metrics_path: str):
    """
    Train a sentiment analysis model using a pipeline that includes
    preprocessing, vectorization, and classification.
    """
    data = pd.read_pickle(input_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)
    print(f"Accuracy: {acc:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model and output metrics")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the preprocessed data pickle (contains X_train, X_test, y_train, y_test)"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path where the trained model will be saved (e.g. models/model.pkl)"
    )
    parser.add_argument(
        "--metrics", "-e",
        required=True,
        help="Path where the metrics JSON will be written (e.g. metrics/accuracy.json)"
    )
    args = parser.parse_args()
    train(args.input, args.model, args.metrics)