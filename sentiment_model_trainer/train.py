import pandas as pd
import os
import joblib
import json
import argparse

from libml.preprocessing import build_pipeline
from sklearn.metrics import accuracy_score

def train(input_path: str, model_path: str, metrics_path: str, vectorizer_path: str = None ):
    """
    Train a sentiment analysis model using a pipeline that includes
    preprocessing, vectorization, and classification.
    """
    # Ensure the output directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    data = pd.read_pickle(input_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    if vectorizer_path:
        pipeline = build_pipeline(vectorizer_text = False)
    else:
        pipeline = build_pipeline(vectorizer_text = True)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    #Save 2 models if vectorizer is provided
    if vectorizer_path:
        print("Vectorizer provided, saving two models: one with and one without vectorizer.")
        joblib.dump(pipeline, model_path.replace(".pkl","-without-vectorizer.pkl")) #model without vectorizer

        # Load the vectorizer and insert it into the pipeline for 2nd model
        vectorizer = joblib.load(vectorizer_path)
        pipeline.steps.insert(0, ("vectorizer", vectorizer))
        joblib.dump(pipeline, model_path) #model with vectorizer
    else:
        joblib.dump(pipeline, model_path)

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
        "--vectorizer", "-v",
        required=False,
        help="Path to the vectorizer pickle (e.g. artifacts/c1_BoW_Sentiment_Model.pkl)"
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
    train(args.input, args.model, args.metrics, args.vectorizer)