import os
import tempfile
import pickle
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sentiment_model_trainer.train import train


@pytest.fixture
def fake_preprocessed_data():
    data = {
        "X_train": ["good food", "the food was great", "such a bad service", "disappointed with the pasta and service"],
        "X_test": ["glad we found this place", "never been more insulted "],
        "y_train": [1, 1, 0, 0],
        "y_test": [1, 0]
    }
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        with open(tmp.name, "wb") as f:
            pickle.dump(data, f)
        yield tmp.name
    os.remove(tmp.name)


@pytest.fixture
def fake_vectorizer():
    vec = CountVectorizer()
    vec.fit(["good", "bad", "excellent", "terrible", "plot", "film", "acting", "movie"])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        joblib.dump(vec, tmp.name)
        yield tmp.name
    os.remove(tmp.name)


@pytest.fixture
def fake_preprocessed_data_with_vectorizer(fake_preprocessed_data, fake_vectorizer):

    indata = pd.read_pickle(fake_preprocessed_data)
    X_train, X_test = indata["X_train"], indata["X_test"]
    y_train, y_test = indata["y_train"], indata["y_test"]

    with open(fake_vectorizer, "rb") as f:
        vec = pickle.load(f)

    X_train = vec.transform(X_train).toarray()
    X_test = vec.transform(X_test).toarray()

    data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        with open(tmp.name, "wb") as f:
            pickle.dump(data, f)
        yield tmp.name
    os.remove(tmp.name)



def test_train_without_vectorizer(fake_preprocessed_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        metrics_path = os.path.join(tmpdir, "metrics.json")

        train(
            input_path=fake_preprocessed_data,
            model_path=model_path,
            metrics_path=metrics_path,
            vectorizer_path=None
        )

        # Check model saved
        assert os.path.exists(model_path)
        model = joblib.load(model_path)
        assert isinstance(model, Pipeline)

        # Check metrics
        assert os.path.exists(metrics_path)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0




def test_train_with_vectorizer(fake_preprocessed_data_with_vectorizer, fake_vectorizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        metrics_path = os.path.join(tmpdir, "metrics.json")
        model_no_vec_path = model_path.replace(".pkl", "-without-vectorizer.pkl")


        train(
            input_path=fake_preprocessed_data_with_vectorizer,
            model_path=model_path,
            metrics_path=metrics_path,
            vectorizer_path=fake_vectorizer
        )

        # Check both models saved
        assert os.path.exists(model_path)
        assert os.path.exists(model_no_vec_path)

        model_with_vec = joblib.load(model_path)
        assert isinstance(model_with_vec, Pipeline)
        assert model_with_vec.named_steps.get("vectorizer") is not None

        model_without_vec = joblib.load(model_no_vec_path)
        assert isinstance(model_without_vec, Pipeline)
        assert "vectorizer" not in dict(model_without_vec.named_steps)

        # Check metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
