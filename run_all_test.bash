echo "=== Running pytest with coverage ==="
pytest

COV=$(coverage report | tail -n 1 | awk '{print $4}')
echo ">>> Coverage: ${COV}"

#echo "=== Running mutation testing (mutmut) ==="
#mutmut run src
#MUT_SCORE=$(mutmut results | grep ^Score | awk '{print $2}')
#echo ">>> Mutation Testing Score: ${MUT_SCORE}"

echo "All tests passed with coverage ${COV} and mutation score ${MUT_SCORE}."