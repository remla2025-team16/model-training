import pandas as pd
import os
import joblib

from libml.preprocessing import build_pipeline

data_path = "data/a1_RestaurantReviews_HistoricDump.tsv"
df = pd.read_csv(data_path, sep="\t")

X = df["Review"]
y = df["Liked"]

pipeline = build_pipeline()

pipeline.fit(X, y)

os.makedirs("models", exist_ok=True)
model_path = "models/sentiment-model.pkl"
joblib.dump(pipeline, model_path)

print(f"Model saved to: {model_path}")

# Test the model
if __name__ == "__main__":
    model_path = "models/sentiment-model.pkl"
    pipeline = joblib.load(model_path)
    test_data = [
        "The food was great and the service was excellent!",
        "I didn't like the ambiance of the restaurant.",
        "The staff were rude and unhelpful.",
        "I had a wonderful experience overall."
    ]

    predictions = pipeline.predict(test_data)
    print("Predictions for test data:", predictions)