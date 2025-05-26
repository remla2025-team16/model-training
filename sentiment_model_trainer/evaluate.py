import argparse
import json
import os
import pickle

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(
        model_path: str,
        data_path: str,
        output_path: str
) -> str:
    """
    Evaluate the trained model on the tests set, save selected metrics, and copy the model artifact.
    """
    model = joblib.load(model_path)

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {recall:.4f}")
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on tests set and save metrics"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the trained model pickle (e.g. models/model.pkl)"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to preprocessed data pickle (contains X_test, y_test)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path where evaluation metrics JSON will be saved (e.g. metrics/evaluation.json)"
    )
    args = parser.parse_args()
    evaluate(args.model, args.data, args.output)
