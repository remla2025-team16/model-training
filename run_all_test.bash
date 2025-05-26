echo "=== Running pytest with coverage ==="
pytest --cov=src --cov-report=term-missing > test_report.txt
COV=$(coverage report | tail -n 1 | awk '{print $4}')
echo ">>> Coverage: ${COV}"

echo "=== Running mutation testing (mutmut) ==="
mutmut run src
MUT_SCORE=$(mutmut results | grep ^Score | awk '{print $2}')
echo ">>> Mutation Testing Score: ${MUT_SCORE}"

BENCH_LINE=$(grep 'test_inference_latency' test_report.txt)
MIN_LAT=$(echo $BENCH_LINE | awk '{print $2}' | tr -d ',')
MAX_LAT=$(echo $BENCH_LINE | awk '{print $3}' | tr -d ',')
MEAN_LAT=$(echo $BENCH_LINE | awk '{print $4}' | tr -d ',')
echo ">>> Latency - Min: $MIN_LAT µs, Max: $MAX_LAT µs, Mean: $MEAN_LAT µs"

if [ ! -d "metrics" ]; then
  echo "Creating metrics directory..."
  mkdir "metrics"
else
  echo "Metrics directory already exists."
fi

cat <<EOF > metrics/test_metrics.json
{
  "coverage": "${COV}",
  "mutation_score": "${MUT_SCORE}",
  "latency": {
    "min_us": ${MIN_LAT},
    "max_us": ${MAX_LAT},
    "mean_us": ${MEAN_LAT}
  }
}
EOF

rm test_report.txt