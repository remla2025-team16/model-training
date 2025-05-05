import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data_path = "../data/a1_RestaurantReviews_HistoricDump.tsv"
df = pd.read_csv(data_path, sep='\t')

X = df['text']
y = df['label']

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=200))
])

pipeline.fit(X, y)

os.makedirs("../models", exist_ok=True)
model_path = "../models/sentiment-model-v1.0.0.pkl"
joblib.dump(pipeline, model_path)

print(f"Model trained and saved to: {model_path}")
