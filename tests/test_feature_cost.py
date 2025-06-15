import time
from tests.utils_test import return_model


pipeline = return_model()


def test_inference_time_large_input():
    text = "very " * 1000 + "good"
    start = time.time()
    pipeline.predict([text])[0]
    elapsed = time.time() - start
    assert elapsed < 2.0
