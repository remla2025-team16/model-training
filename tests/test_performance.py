# tests/test_performance.py
import os

import psutil
import pytest

from tests.utils_test import return_model

pipeline = return_model()


@pytest.mark.benchmark(group="inference_latency")
def test_inference_latency(benchmark):
    """
    Ensure that works predict_sentiment() on a typical sentence and benchmark its latency.
    """

    def infer():
        return pipeline.predict(["The food was fantastic and the service was quick."])[0]

    result = benchmark(infer)

    assert result > 0.5, "Prediction is not correct"


def test_memory_usage_during_inference():
    """
    Ensure memory consumption never spikes above 200 MB.
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss  # in bytes

    for _ in range(10):
        pipeline.predict(["The ambiance was lovely."])[0]
    mem_after = process.memory_info().rss

    used = (mem_after - mem_before) / (1024 * 1024)
    assert used < 50, f"Memory bump too high: {used:.1f} MB"

    # some random number, but not too high. Could be tune
