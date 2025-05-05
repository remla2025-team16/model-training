import pandas as pd
import os
import joblib

from libml.preprocessing import build_pipeline

data_path = "../data/a1_RestaurantReviews_HistoricDump.tsv"
df = pd.read_csv(data_path, sep="\t")

X = df["text"]
y = df["label"]

pipeline = build_pipeline()

pipeline.fit(X, y)

os.makedirs("../models", exist_ok=True)
model_path = "../models/sentiment-model-v1.0.0.pkl"
joblib.dump(pipeline, model_path)

print(f"Model saved to: {model_path}")
