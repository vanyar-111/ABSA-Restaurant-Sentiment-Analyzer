"""
aspect_extractor.py
===================
Extracts candidate aspect terms from review text using:
1. Part-of-Speech (POS) pattern chunking (Noun chunks: JJ* NN+)
2. Domain aspect lexicon filtering mined from SemEval-2014

Academic Explanation:
---------------------
In Aspect-Based Sentiment Analysis (ABSA), aspect extraction is the subtask
of identifying words or phrases representing target entities or attributes
of a restaurant (e.g., 'food', 'service', 'crust', 'wait staff', 'ambiance').

Classical approaches:
- Rule-based Syntactic Chunking: In English, restaurant aspects are overwhelmingly
  nouns (NN, NNS) or compound noun phrases preceded by optional modifiers
  (e.g., [attentive/JJ] [wait/NN] [staff/NN]).
- Domain Lexicon Validation: To eliminate non-aspect common nouns (such as 'time',
  'day', 'person'), candidates are matched or ranked against a domain vocabulary
  derived from the SemEval restaurant dataset.
"""

import re
import nltk
from typing import List, Dict, Set, Tuple, Optional
from src.preprocessor import clean_text, tokenize, pos_tag_tokens


# Sentiment adjectives that should NOT be part of the aspect name itself
SENTIMENT_ADJECTIVES: Set[str] = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "delicious",
    "bad", "terrible", "horrible", "awful", "poor", "overpriced", "expensive",
    "cheap", "slow", "fast", "quick", "pleasant", "nice", "clean", "dirty",
    "average", "decent", "rude", "friendly", "attentive", "cozy", "noisy", "loud"
}

# Default fallback restaurant aspects seed lexicon
DEFAULT_RESTAURANT_ASPECTS: Set[str] = {
    "food", "service", "staff", "waiter", "waitress", "hostess", "management",
    "ambiance", "atmosphere", "decor", "interior", "environment", "music",
    "price", "prices", "bill", "cost", "value", "menu", "portion", "portions",
    "drink", "drinks", "wine", "beer", "cocktail", "cocktails", "bar",
    "pizza", "pasta", "sushi", "burger", "steak", "chicken", "fish", "bread",
    "dessert", "salad", "appetizer", "seafood", "cheese", "sauce", "crust",
    "table", "seat", "seating", "place", "location", "reservation", "meal",
    "dinner", "lunch", "breakfast", "dining", "experience", "wait", "quality"
}

# Stopwords that should never be identified as standalone aspects
EXCLUDED_ASPECT_TOKENS: Set[str] = {
    "i", "we", "you", "they", "he", "she", "it", "one", "everyone", "someone",
    "thing", "things", "lot", "lots", "way", "bit", "kind", "part", "nothing",
    "anything", "everything", "times", "time", "day", "night", "week", "year",
    "restaurant", "restaurants", "special", "taste"
}


class AspectExtractor:
    """
    Extracts restaurant aspect terms using POS syntactic chunking
    and domain lexicon validation.
    """

    def __init__(self, domain_lexicon: Optional[Set[str]] = None):
        """
        Initializes the extractor with a domain lexicon.
        If domain_lexicon is None, defaults to DEFAULT_RESTAURANT_ASPECTS.
        """
        self.lexicon = set(domain_lexicon) if domain_lexicon else set(DEFAULT_RESTAURANT_ASPECTS)
        # Add singular and plural variations
        expanded = set()
        for term in self.lexicon:
            expanded.add(term.lower())
            if term.endswith("s"):
                expanded.add(term[:-1].lower())
            else:
                expanded.add((term + "s").lower())
        self.lexicon.update(expanded)

        # Define POS chunk grammar: Nouns and compound nouns
        self.grammar = r"""
            ASPECT: {<NN|NNS|NNP|NNPS>+}
        """
        try:
            self.chunk_parser = nltk.RegexpParser(self.grammar)
        except Exception:
            self.chunk_parser = None

    def extract_candidates_pos(self, text: str) -> List[Dict[str, any]]:
        """
        Extracts noun-phrase candidates using NLTK POS chunking.
        Returns list of dicts with 'term', 'pos_tag', 'start_char', 'end_char'.
        """
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        tagged = pos_tag_tokens(tokens)

        candidates = []
        if self.chunk_parser is not None:
            tree = self.chunk_parser.parse(tagged)
            for subtree in tree.subtrees(filter=lambda t: t.label() == "ASPECT"):
                words = [w for w, pos in subtree.leaves()]
                phrase = " ".join(words).strip()
                # Extract the core head noun or phrase
                if phrase.lower() not in EXCLUDED_ASPECT_TOKENS and len(phrase) > 1:
                    candidates.append({
                        "term": phrase,
                        "raw_words": words,
                        "pos_tags": [pos for w, pos in subtree.leaves()]
                    })
        else:
            # Fallback simple contiguous noun grouping
            current = []
            for word, pos in tagged:
                if pos.startswith("NN"):
                    current.append(word)
                else:
                    if current:
                        phrase = " ".join(current)
                        if phrase.lower() not in EXCLUDED_ASPECT_TOKENS:
                            candidates.append({"term": phrase, "raw_words": current, "pos_tags": ["NN"]*len(current)})
                        current = []
            if current:
                phrase = " ".join(current)
                if phrase.lower() not in EXCLUDED_ASPECT_TOKENS:
                    candidates.append({"term": phrase, "raw_words": current, "pos_tags": ["NN"]*len(current)})

        return candidates

    def extract_aspects(self, text: str) -> List[str]:
        """
        Main extraction method:
        Given raw review text, identifies the high-confidence restaurant aspects.
        Prioritizes:
        1. Multi-word and single-word matches against domain lexicon
        2. Valid noun chunk candidates whose head noun belongs to restaurant domain

        Returns:
        --------
        List[str]:
            Clean list of distinct aspect terms detected in text order.
        """
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        extracted: List[Tuple[int, str]] = []  # (start_index, aspect_term)

        # Pass 1: Direct multi-word lexicon matching (e.g. "wait staff", "beer selection")
        for lex_term in sorted(self.lexicon, key=lambda x: len(x), reverse=True):
            if " " in lex_term:
                pattern = r"\b" + re.escape(lex_term) + r"\b"
                for match in re.finditer(pattern, text_lower):
                    extracted.append((match.start(), text[match.start():match.end()]))

        # Pass 2: POS Chunk candidate filtering
        pos_candidates = self.extract_candidates_pos(text)
        for cand in pos_candidates:
            term = cand["term"]
            term_lower = term.lower()
            words = term_lower.split()

            # Check if candidate itself or any constituent noun is in lexicon
            matched = False
            if term_lower in self.lexicon:
                matched = True
            else:
                for w in words:
                    if w in self.lexicon and w not in EXCLUDED_ASPECT_TOKENS:
                        matched = True
                        break

            if matched:
                # Find start position in text
                pattern = r"\b" + re.escape(term) + r"\b"
                match = re.search(pattern, text, re.IGNORECASE)
                start_pos = match.start() if match else len(extracted)
                extracted.append((start_pos, term))

        # Pass 3: Single-word lexicon sweep for any missed core keywords (e.g., 'food', 'service')
        for word in tokenize(text):
            word_lower = word.lower()
            if word_lower in self.lexicon and word_lower not in EXCLUDED_ASPECT_TOKENS:
                pattern = r"\b" + re.escape(word) + r"\b"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted.append((match.start(), word))

        # Sort by position in review text
        extracted.sort(key=lambda x: x[0])

        # Deduplicate while preserving order and longest matching spans
        final_aspects = []
        for _, term in extracted:
            clean_term = term.strip()
            # If sub-string of already added term (e.g. 'food' inside 'thai food'), skip or keep specific
            already_covered = False
            for existing in final_aspects:
                if clean_term.lower() == existing.lower() or (
                    clean_term.lower() in existing.lower().split()
                ):
                    already_covered = True
                    break
            if not already_covered and clean_term.lower() not in EXCLUDED_ASPECT_TOKENS:
                final_aspects.append(clean_term)

        return final_aspects


if __name__ == "__main__":
    extractor = AspectExtractor()
    test_cases = [
        "The food was amazing but the service was extremely slow.",
        "Delicious thin crust pizza and attentive wait staff, though the wine list is small.",
        "Great atmosphere and affordable prices.",
        "The chicken was raw and the soup was freezing cold."
    ]
    for tc in test_cases:
        print(f"\nReview: '{tc}'")
        aspects = extractor.extract_aspects(tc)
        print("Detected Aspects:", aspects)
