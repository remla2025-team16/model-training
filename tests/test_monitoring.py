import requests
import re


def test_prometheus_metrics_format():
    response = requests.get("http://localhost:8080/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["Content-Type"]
    assert "# HELP" in response.text


def test_custom_metrics_present():
    response = requests.get("http://localhost:8080/metrics")
    content = response.text
    expected_metrics = [
        "webapp_predictions_total",
        "webapp_response_latency_seconds",
        "webapp_ram_usage_bytes"
    ]
    for metric in expected_metrics:
        assert re.search(rf"^{metric}(\{{.*\}})?\s", content, re.MULTILINE)
