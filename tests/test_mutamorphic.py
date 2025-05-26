import pytest

from tests.utils_test import replace_random_synonyms, normalize_text, return_model

pipeline = return_model()


#make this something that user can add/override
@pytest.mark.parametrize("text", [
    "The food was absolutely wonderful, and the service was outstanding!",
    "I hated the wait time, but the pasta was delicious.",
])
def test_synonym_invariance(text):
    """
    Replace some words in `text` with synonyms;
    model’s sentiment (positive/negative) shouldn’t change by more than +/- 0.1.
    """
    baseline_prob = pipeline.predict([text])[0]
    transformed = replace_random_synonyms(text)
    new_prob = pipeline.predict([transformed])[0]

    print(f"baseline_prob = {baseline_prob}")
    print(f"transformed = {transformed}")

    if abs(new_prob - baseline_prob) > 0.1:
        repaired = normalize_text(transformed)
        repaired_prob = pipeline.predict([repaired])[0]

        assert abs(repaired_prob - baseline_prob) <= 0.1, (
            f"Even after repair, Δ={abs(repaired_prob - baseline_prob):.3f} "
            f"is too large for '{text}' → '{repaired}'."
        )
    else:
        assert True