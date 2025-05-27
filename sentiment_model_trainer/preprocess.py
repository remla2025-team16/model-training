import argparse
import os
import pickle

from libml.preprocessing import data_preprocessing


def process(
        data_path: str,
        preprocessed_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        artifacts_dir: str = "artifacts",
        vectorizer_filename: str = "c1_BoW_Sentiment_Model.pkl",
) -> str:
    """
    Call data_preprocessing to generate train/tests splits and save both the
    fitted CountVectorizer and the split datasets as pickle files.

    Args:
        data_path: Path to the raw TSV dataset with 'Review' column.
        preprocessed_path: Path to the preprocessed dataset.
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for reproducibility.
        artifacts_dir: Directory to store artifacts.
        vectorizer_filename: Filename for the saved CountVectorizer.


    Returns:
        Path to the pickle file containing the preprocessed data dict.
    """

    # ensure artifacts directory exists
    os.makedirs(artifacts_dir, exist_ok=True)

    vectorizer_path = os.path.join(artifacts_dir, vectorizer_filename)

    # generate data and save vectorizer
    x_train, x_test, y_train, y_test = data_preprocessing(
        filepath=data_path,
        test_size=test_size,
        random_state=random_state,
        vectorizer_output=vectorizer_path,
        vectorizer_input=None
    )

    # save preprocessed splits
    with open(preprocessed_path, "wb") as f:
        pickle.dump({
            "X_train": x_train,
            "X_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        }, f)

    print(f"Preprocessed data saved to: {preprocessed_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

    return preprocessed_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="data_path", required=True)
    parser.add_argument("--output", dest="preprocessed_path", required=True)

    args = parser.parse_args()
    process(
        data_path=args.data_path,
        preprocessed_path=args.preprocessed_path,
    )


if __name__ == "__main__":
    main()
