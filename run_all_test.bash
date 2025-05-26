echo "=== Running pytest with coverage ==="
pytest --cov=sentiment_model_trainer --cov-report=term-missing | tee test_report.txt
COV=$(coverage report | tail -n 1 | awk '{print $4}')
echo ">>> Coverage: ${COV}"

echo "=== Running mutation testing (mutmut) ==="
mutmut run  | tee mutmut_report.txt

PASSED=$(cat mutmut_report.txt | tail -n 2 | head -n 1 | grep -o "🎉 [0-9]*"| tail -n 1 | awk '{print $2}' )
TOTAL=$(cat mutmut_report.txt | tail -n 2 | head -n 1 | grep -o "0/[0-9]*" | cut -c 3-)
MUT_SCORE=$(awk -v p="$PASSED" -v t="$TOTAL" 'BEGIN { printf "%.2f", (p/t)*100 }')

echo ">>> Mutation Passed: ${MUT_SCORE}%"

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

# Clean up temporary files
rm test_report.txt
rm mutmut_report.txt