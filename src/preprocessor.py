"""
preprocessor.py
===============
Module providing core Natural Language Processing (NLP) preprocessing functions:
- Contraction expansion (vital for negation preservation)
- Tokenization
- Stopword management (preserving negation and contrastive markers)
- Part-Of-Speech (POS) tagging using NLTK

Academic Explanation:
---------------------
In sentiment analysis, standard aggressive preprocessing (like deleting all stopwords)
often breaks meaning by removing negation words such as 'not', 'never', 'hardly'
and contrastive conjunctions such as 'but', 'however'.

For Aspect-Based Sentiment Analysis (ABSA):
1. We expand contractions ('wasn\'t' -> 'was not') so negations become explicit tokens.
2. We retain punctuation and conjunctions for syntactic boundary detection.
3. We perform POS Tagging using the Penn Treebank tagset to identify nouns (NN, NNS)
   which serve as aspect candidates, and adjectives (JJ) which convey sentiment.
"""

import re
import nltk
from typing import List, Tuple, Set


# Helper to ensure required NLTK corpora are downloaded quietly
def ensure_nltk_resources():
    """
    Downloads necessary NLTK corpora if not already present on the machine.
    """
    resources = [
        "tokenizers/punkt",
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
        "corpora/stopwords"
    ]
    for res in resources:
        try:
            nltk.data.find(res)
        except (LookupError, IndexError):
            name = res.split("/")[-1]
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


ensure_nltk_resources()


# Common English contractions dictionary
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "n't": " not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "it's": "it is",
    "that's": "that is",
    "i'm": "i am",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are"
}

# Negation and contrastive markers that must NEVER be removed as stopwords
PRESERVED_SENTIMENT_WORDS: Set[str] = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "barely", "scarcely",
    "but", "however", "although", "yet", "though", "except", "while",
    "very", "extremely", "really", "too", "so", "quite"
}


def expand_contractions(text: str) -> str:
    """
    Expands conversational contractions to preserve polarity signals.
    Example: "The food wasn't good" -> "The food was not good"
    """
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b", re.IGNORECASE)
    def replace(match):
        return CONTRACTION_MAP[match.group(0).lower()]
    return pattern.sub(replace, text)


def clean_text(text: str) -> str:
    """
    Cleans review text:
    - Normalizes multiple whitespace characters
    - Expands contractions
    - Keeps punctuation necessary for clause boundary splitting
    """
    if not isinstance(text, str):
        return ""
    text = expand_contractions(text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into words and punctuation tokens using NLTK word_tokenize.
    """
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        # Fallback regex tokenizer if NLTK punkt encounters an issue
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens


def pos_tag_tokens(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Performs Part-Of-Speech (POS) tagging on a list of tokens.
    Returns list of (token, pos_tag) pairs.
    Penn Treebank tags:
      NN: Noun, singular (e.g., 'table', 'food')
      NNS: Noun, plural (e.g., 'drinks', 'prices')
      NNP: Proper noun (e.g., 'Sushi', 'Manhattan')
      JJ: Adjective (e.g., 'delicious', 'slow')
      RB: Adverb (e.g., 'extremely', 'not')
    """
    try:
        return nltk.pos_tag(tokens)
    except Exception:
        # Fallback simple rule if tagger not loaded
        return [(t, "NN" if t.isalnum() else "PUNCT") for t in tokens]


def get_sentiment_stopwords() -> Set[str]:
    """
    Returns NLTK English stopwords minus critical sentiment, negation,
    and contrastive conjunction words.
    """
    try:
        from nltk.corpus import stopwords
        base_stops = set(stopwords.words("english"))
    except Exception:
        base_stops = {"the", "a", "an", "in", "on", "at", "by", "for", "with", "about", "against", "into"}
    
    # Remove words essential for sentiment & clause logic
    return base_stops - PRESERVED_SENTIMENT_WORDS


if __name__ == "__main__":
    sample = "The food was amazing, but the service wasn't fast at all!"
    cleaned = clean_text(sample)
    toks = tokenize(cleaned)
    tagged = pos_tag_tokens(toks)
    print("Original:", sample)
    print("Cleaned:", cleaned)
    print("Tokens:", toks)
    print("POS Tags:", tagged[:8])
