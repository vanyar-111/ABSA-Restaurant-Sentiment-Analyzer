import re
from typing import List, Tuple, Optional
from src.preprocessor import clean_text, tokenize


ADVERSATIVE_CONJUNCTIONS = [
    "but", "however", "although", "yet", "though", "except", "while", "whereas"
]


def split_into_clauses(text: str) -> List[str]:
  
    conj_pattern = r"(?<=\s)(?:" + "|".join(ADVERSATIVE_CONJUNCTIONS) + r")(?=\s)"
    punct_pattern = r"[;,\.\!\?]\s+"

    delimited = re.sub(conj_pattern, " <SPLIT> ", text, flags=re.IGNORECASE)
    delimited = re.sub(punct_pattern, " <SPLIT> ", delimited)

    clauses = [c.strip() for c in delimited.split("<SPLIT>") if c.strip()]
    return clauses if clauses else [text]


def extract_aspect_context(
    sentence: str,
    aspect_term: str,
    window_size: int = 6
) -> str:
    cleaned_sent = clean_text(sentence)
    aspect_lower = aspect_term.lower().strip()

    clauses = split_into_clauses(cleaned_sent)

    target_clause = None
    for clause in clauses:
        if re.search(r"\b" + re.escape(aspect_lower) + r"\b", clause.lower()):
            target_clause = clause
            break

    if target_clause is None:
        target_clause = cleaned_sent

    tokens = tokenize(target_clause)
    tokens_lower = [t.lower() for t in tokens]

    aspect_toks = tokenize(aspect_lower)
    match_idx = -1
    for i in range(len(tokens_lower)):
        if tokens_lower[i:i + len(aspect_toks)] == aspect_toks:
            match_idx = i
            break

    if match_idx != -1:
   
        start = max(0, match_idx - window_size)
        end = min(len(tokens), match_idx + len(aspect_toks) + window_size)
        windowed_tokens = tokens[start:end]
        context_str = " ".join(windowed_tokens)
    else:
        context_str = target_clause

    return context_str


def build_aspect_context_dataset(
    sentences: List[str],
    aspect_terms: List[str]
) -> List[str]:
    return [
        extract_aspect_context(s, a)
        for s, a in zip(sentences, aspect_terms)
    ]


if __name__ == "__main__":
    review = "The food was amazing but the service was extremely slow."
    ctx_food = extract_aspect_context(review, "food")
    ctx_service = extract_aspect_context(review, "service")
    print("Original Review:", review)
    print("Context for 'food':", ctx_food)
    print("Context for 'service':", ctx_service)
