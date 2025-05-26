import random
import nltk
from nltk.corpus import wordnet
import unicodedata


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
