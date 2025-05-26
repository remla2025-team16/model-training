import os
import tempfile
import pickle
import pytest
import pandas as pd

from src.preprocess import process


@pytest.fixture
def sample_tsv():
    data = pd.DataFrame({
        "Review": ["Great product!", "Terrible experience.", "Average performance", "Loved it", "Hated it"],
        "Sentiment": [1, 0, 1, 1, 0]
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv", mode='w') as tmp:
        data.to_csv(tmp.name, sep='\t', index=False)
        yield tmp.name
    os.remove(tmp.name)

def test_process_function(sample_tsv):
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessed_path = os.path.join(tmpdir, "preprocessed_data.pkl")
        vectorizer_filename = "vectorizer.pkl"

        # Call the function
        returned_path = process(
            data_path=sample_tsv,
            preprocessed_path=preprocessed_path,
            artifacts_dir=tmpdir,
            vectorizer_filename=vectorizer_filename
        )


        assert returned_path == preprocessed_path
        assert os.path.exists(preprocessed_path)
        assert os.path.exists(os.path.join(tmpdir, vectorizer_filename))

        with open(preprocessed_path, "rb") as f:
            data = pickle.load(f)

        assert "X_train" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "y_test" in data

        assert len(data["X_train"]) > 0
        assert len(data["X_test"]) > 0

