import os
import random
import unicodedata

import joblib
from nltk.corpus import wordnet

MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment-model.pkl")

if not os.path.isfile(MODEL_PATH):
    MODEL_PATH = "/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/Year1/q4/CS4295/model-training/models/sentiment-model.pkl"

print(MODEL_PATH)

MODEL_URL = os.getenv("MODEL_URL", None)
if MODEL_URL and not os.path.isfile(MODEL_PATH):
    import requests

    resp = requests.get(MODEL_URL)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(resp.content)

pipeline = joblib.load(MODEL_PATH)


def return_model():
    return pipeline


def get_synonyms(word: str):
    synsets = wordnet.synsets(word)
    synonyms = set()
    for s in synsets:
        for lemma in s.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)


def replace_random_synonyms(sentence: str, max_replacements: int = 2) -> str:
    tokens = sentence.split()
    candidate_indices = [i for i, t in enumerate(tokens) if get_synonyms(t)]
    if not candidate_indices:
        return sentence  # no synonyms found
    num_swaps = min(len(candidate_indices), random.randint(1, max_replacements))
    indices_to_swap = random.sample(candidate_indices, num_swaps)
    for idx in indices_to_swap:
        syns = get_synonyms(tokens[idx])
        if syns:
            tokens[idx] = random.choice(syns)
    return " ".join(tokens)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    return " ".join(text.strip().split())
