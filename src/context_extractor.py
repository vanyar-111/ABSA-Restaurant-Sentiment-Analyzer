"""
context_extractor.py
====================
Extracts local, aspect-specific context from review sentences.

Academic Explanation:
---------------------
Why is Aspect-Context Extraction the core technical challenge of ABSA?

Consider the benchmark sentence:
  "The food was amazing but the service was extremely slow."

If a standard sentiment model examines the whole sentence for 'service',
it sees conflicting words: 'amazing' and 'slow'.
Without context isolation, the model confuses which adjective modifies which noun.

Technique Implemented:
1. Clause-Boundary Splitting:
   We identify adversative connectives ('but', 'however', 'although', 'yet',
   'though', 'except', 'while') and major punctuation marks (',', ';', '.')
   to partition the review into independent semantic clauses.
2. Target Clause Retrieval:
   For any given aspect term, we locate the specific clause enclosing it.
3. Windowed Context Extraction:
   If clauses are long or ambiguous, we apply a localized window of $k$ tokens
   (default $k=6$) around the aspect term while respecting clause boundaries.
4. Aspect-Aware Representation:
   The extracted context is formatted as:
     `[aspect_term] + " " + [clause_context]`
   This strongly ties the aspect term to its modifying adjectives and adverbs
   during TF-IDF feature extraction.
"""

import re
from typing import List, Tuple, Optional
from src.preprocessor import clean_text, tokenize


# Adversative connectives indicating shift in sentiment
ADVERSATIVE_CONJUNCTIONS = [
    "but", "however", "although", "yet", "though", "except", "while", "whereas"
]


def split_into_clauses(text: str) -> List[str]:
    """
    Splits a complex sentence into constituent clauses based on
    adversative conjunctions and strong punctuation delimiters.
    """
    # Create regex pattern splitting on connectives with word boundaries or punctuation
    conj_pattern = r"(?<=\s)(?:" + "|".join(ADVERSATIVE_CONJUNCTIONS) + r")(?=\s)"
    punct_pattern = r"[;,\.\!\?]\s+"

    # First split by conjunctions (preserving boundary content)
    # We replace conjunctions with a delimiter
    delimited = re.sub(conj_pattern, " <SPLIT> ", text, flags=re.IGNORECASE)
    delimited = re.sub(punct_pattern, " <SPLIT> ", delimited)

    clauses = [c.strip() for c in delimited.split("<SPLIT>") if c.strip()]
    return clauses if clauses else [text]


def extract_aspect_context(
    sentence: str,
    aspect_term: str,
    window_size: int = 6
) -> str:
    """
    Extracts the local sentiment context for a specific aspect term.

    Parameters:
    -----------
    sentence : str
        The full review sentence.
    aspect_term : str
        The aspect target (e.g. 'food', 'service').
    window_size : int
        Token radius around aspect if clause is broad.

    Returns:
    --------
    str:
        Aspect-context string combining the aspect term and its local modifiers.
    """
    cleaned_sent = clean_text(sentence)
    aspect_lower = aspect_term.lower().strip()

    # Step 1: Divide sentence into clauses
    clauses = split_into_clauses(cleaned_sent)

    # Step 2: Find the clause containing this aspect
    target_clause = None
    for clause in clauses:
        if re.search(r"\b" + re.escape(aspect_lower) + r"\b", clause.lower()):
            target_clause = clause
            break

    # If not found directly in split clauses, use entire cleaned sentence
    if target_clause is None:
        target_clause = cleaned_sent

    # Step 3: Apply token window within the target clause
    tokens = tokenize(target_clause)
    tokens_lower = [t.lower() for t in tokens]

    # Find the position of the aspect term in tokens
    aspect_toks = tokenize(aspect_lower)
    match_idx = -1
    for i in range(len(tokens_lower)):
        if tokens_lower[i:i + len(aspect_toks)] == aspect_toks:
            match_idx = i
            break

    if match_idx != -1:
        # Window of window_size tokens before and after
        start = max(0, match_idx - window_size)
        end = min(len(tokens), match_idx + len(aspect_toks) + window_size)
        windowed_tokens = tokens[start:end]
        context_str = " ".join(windowed_tokens)
    else:
        context_str = target_clause

    # Step 4: Return Aspect-Targeted Context Representation
    # Prepends aspect term to ensure high aspect-specific weighting in TF-IDF
    return f"{aspect_term} {context_str}"


def build_aspect_context_dataset(
    sentences: List[str],
    aspect_terms: List[str]
) -> List[str]:
    """
    Batch helper to convert paired sentences and aspect terms into context features.
    """
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
