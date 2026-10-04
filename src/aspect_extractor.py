"""
aspect_extractor.py
===================
Extracts candidate aspect terms from review text using:
1. Part-of-Speech (POS) pattern chunking
2. Domain aspect lexicon filtering
3. Sentiment-adjective filtering
4. Single-word lexicon fallback

In Aspect-Based Sentiment Analysis (ABSA), aspect extraction identifies
entities or attributes that are being discussed in a review, such as:
food, service, crust, wait staff, ambiance, prices, etc.
"""

import re
import nltk
from typing import List, Dict, Set, Tuple, Optional

from src.preprocessor import clean_text, tokenize, pos_tag_tokens


# ---------------------------------------------------------------------------
# Sentiment adjectives
# ---------------------------------------------------------------------------

# These words describe sentiment and should not become part of the
# aspect name itself.
SENTIMENT_ADJECTIVES: Set[str] = {
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "fantastic",
    "delicious",
    "bad",
    "terrible",
    "horrible",
    "awful",
    "poor",
    "overpriced",
    "expensive",
    "cheap",
    "slow",
    "fast",
    "quick",
    "pleasant",
    "nice",
    "clean",
    "dirty",
    "average",
    "decent",
    "rude",
    "friendly",
    "attentive",
    "cozy",
    "noisy",
    "loud",
}


# ---------------------------------------------------------------------------
# Default restaurant aspect lexicon
# ---------------------------------------------------------------------------

DEFAULT_RESTAURANT_ASPECTS: Set[str] = {
    "food",
    "service",
    "staff",
    "waiter",
    "waitress",
    "hostess",
    "management",
    "ambiance",
    "atmosphere",
    "decor",
    "interior",
    "environment",
    "music",
    "price",
    "prices",
    "bill",
    "cost",
    "value",
    "menu",
    "portion",
    "portions",
    "drink",
    "drinks",
    "wine",
    "beer",
    "cocktail",
    "cocktails",
    "bar",
    "pizza",
    "pasta",
    "sushi",
    "burger",
    "steak",
    "chicken",
    "fish",
    "bread",
    "dessert",
    "salad",
    "appetizer",
    "seafood",
    "cheese",
    "sauce",
    "crust",
    "table",
    "seat",
    "seating",
    "place",
    "location",
    "reservation",
    "meal",
    "dinner",
    "lunch",
    "breakfast",
    "dining",
    "experience",
    "wait",
    "quality",
}


# ---------------------------------------------------------------------------
# Excluded generic terms
# ---------------------------------------------------------------------------

EXCLUDED_ASPECT_TOKENS: Set[str] = {
    "i",
    "we",
    "you",
    "they",
    "he",
    "she",
    "it",
    "one",
    "everyone",
    "someone",
    "thing",
    "things",
    "lot",
    "lots",
    "way",
    "bit",
    "kind",
    "part",
    "nothing",
    "anything",
    "everything",
    "times",
    "time",
    "day",
    "night",
    "week",
    "year",
    "restaurant",
    "restaurants",
    "special",
    "taste",
}


class AspectExtractor:
    """
    Extracts restaurant aspect terms using POS syntactic chunking
    and domain lexicon validation.
    """

    def __init__(
        self,
        domain_lexicon: Optional[Set[str]] = None
    ):
        """
        Initialize the aspect extractor.

        If a domain lexicon is supplied, it is used as the primary
        restaurant-domain vocabulary. Otherwise, the default restaurant
        vocabulary is used.
        """

        if domain_lexicon:
            self.lexicon = set(domain_lexicon)
        else:
            self.lexicon = set(DEFAULT_RESTAURANT_ASPECTS)

        # Normalize lexicon and add simple singular/plural variants.
        expanded = set()

        for term in self.lexicon:
            term = term.lower().strip()

            if not term:
                continue

            expanded.add(term)

            if term.endswith("s"):
                expanded.add(term[:-1])
            else:
                expanded.add(term + "s")

        self.lexicon.update(expanded)

        # POS grammar for contiguous noun phrases.
        self.grammar = r"""
            ASPECT: {<NN|NNS|NNP|NNPS>+}
        """

        try:
            self.chunk_parser = nltk.RegexpParser(self.grammar)
        except Exception:
            self.chunk_parser = None


    # -----------------------------------------------------------------------
    # POS candidate extraction
    # -----------------------------------------------------------------------

    def extract_candidates_pos(
        self,
        text: str
    ) -> List[Dict[str, any]]:
        """
        Extract noun-based candidate aspect phrases using POS tagging.
        """

        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        tagged = pos_tag_tokens(tokens)

        candidates = []

        if self.chunk_parser is not None:

            tree = self.chunk_parser.parse(tagged)

            for subtree in tree.subtrees(
                filter=lambda t: t.label() == "ASPECT"
            ):

                words = [
                    word
                    for word, pos in subtree.leaves()
                ]

                original_words = list(words)

                # Remove sentiment adjectives from the beginning.
                while (
                    words
                    and words[0].lower()
                    in SENTIMENT_ADJECTIVES
                ):
                    words.pop(0)

                if not words:
                    continue

                phrase = " ".join(words).strip()

                if (
                    phrase.lower()
                    not in EXCLUDED_ASPECT_TOKENS
                    and len(phrase) > 1
                ):

                    original_tags = [
                        pos
                        for word, pos in subtree.leaves()
                    ]

                    candidates.append(
                        {
                            "term": phrase,
                            "raw_words": words,
                            "pos_tags": original_tags[
                                -len(words):
                            ],
                        }
                    )

        else:

            # Fallback when the chunk parser cannot be created.
            current = []

            for word, pos in tagged:

                if pos.startswith("NN"):
                    current.append(word)

                else:

                    if current:

                        words = list(current)

                        while (
                            words
                            and words[0].lower()
                            in SENTIMENT_ADJECTIVES
                        ):
                            words.pop(0)

                        if words:

                            phrase = " ".join(words)

                            if (
                                phrase.lower()
                                not in EXCLUDED_ASPECT_TOKENS
                            ):
                                candidates.append(
                                    {
                                        "term": phrase,
                                        "raw_words": words,
                                        "pos_tags": [
                                            "NN"
                                        ] * len(words),
                                    }
                                )

                        current = []

            if current:

                words = list(current)

                while (
                    words
                    and words[0].lower()
                    in SENTIMENT_ADJECTIVES
                ):
                    words.pop(0)

                if words:

                    phrase = " ".join(words)

                    if (
                        phrase.lower()
                        not in EXCLUDED_ASPECT_TOKENS
                    ):
                        candidates.append(
                            {
                                "term": phrase,
                                "raw_words": words,
                                "pos_tags": [
                                    "NN"
                                ] * len(words),
                            }
                        )

        return candidates


    # -----------------------------------------------------------------------
    # Main aspect extraction
    # -----------------------------------------------------------------------

    def extract_aspects(
        self,
        text: str
    ) -> List[str]:
        """
        Extract restaurant aspects from review text.

        Extraction strategy:

        1. Match meaningful multi-word domain phrases.
        2. Extract POS-based noun candidates.
        3. Remove sentiment adjectives from candidates.
        4. Fall back to single-word domain matching.
        5. Prefer longer phrases over individual words.
        """

        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()

        # Each item:
        # (character_start_position, aspect_term)
        extracted: List[Tuple[int, str]] = []


        # ================================================================
        # PASS 1
        # Multi-word domain lexicon matching
        # ================================================================

        for lex_term in sorted(
            self.lexicon,
            key=lambda x: len(x),
            reverse=True
        ):

            if " " not in lex_term:
                continue

            lex_words = lex_term.lower().split()

            # Remove sentiment-bearing words from the beginning.
            while (
                lex_words
                and lex_words[0]
                in SENTIMENT_ADJECTIVES
            ):
                lex_words.pop(0)

            if not lex_words:
                continue

            normalized_term = " ".join(lex_words)

            pattern = (
                r"\b"
                + re.escape(normalized_term)
                + r"\b"
            )

            for match in re.finditer(
                pattern,
                text_lower
            ):

                extracted.append(
                    (
                        match.start(),
                        text[
                            match.start():
                            match.end()
                        ],
                    )
                )


        # ================================================================
        # PASS 2
        # POS noun phrase candidates
        # ================================================================

        pos_candidates = self.extract_candidates_pos(text)

        for candidate in pos_candidates:

            term = candidate["term"].strip()

            if not term:
                continue

            words = term.split()

            # Remove sentiment adjectives from the beginning.
            while (
                words
                and words[0].lower()
                in SENTIMENT_ADJECTIVES
            ):
                words.pop(0)

            if not words:
                continue

            term = " ".join(words)
            term_lower = term.lower()

            if (
                term_lower
                in EXCLUDED_ASPECT_TOKENS
            ):
                continue

            # Accept a candidate if the complete phrase or
            # at least one constituent word belongs to the
            # restaurant domain vocabulary.
            matched = (
                term_lower in self.lexicon
                or any(
                    word.lower() in self.lexicon
                    and word.lower()
                    not in EXCLUDED_ASPECT_TOKENS
                    for word in words
                )
            )

            if not matched:
                continue

            pattern = (
                r"\b"
                + re.escape(term)
                + r"\b"
            )

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                extracted.append(
                    (
                        match.start(),
                        text[
                            match.start():
                            match.end()
                        ],
                    )
                )


        # ================================================================
        # PASS 3
        # Single-word domain lexicon fallback
        # ================================================================

        for word in tokenize(text):

            word_lower = word.lower()

            if (
                word_lower
                not in self.lexicon
            ):
                continue

            if (
                word_lower
                in EXCLUDED_ASPECT_TOKENS
            ):
                continue

            pattern = (
                r"\b"
                + re.escape(word)
                + r"\b"
            )

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if not match:
                continue

            start_pos = match.start()

            # Determine whether this single word is already
            # covered by a longer extracted phrase.
            covered_by_phrase = False

            for (
                existing_start,
                existing_term
            ) in extracted:

                existing_words = (
                    existing_term.lower().split()
                )

                if len(existing_words) <= 1:
                    continue

                existing_end = (
                    existing_start
                    + len(existing_term)
                )

                if (
                    existing_start
                    <= start_pos
                    < existing_end
                ):
                    covered_by_phrase = True
                    break

            # IMPORTANT:
            # This is outside the loop above.
            if not covered_by_phrase:

                extracted.append(
                    (
                        start_pos,
                        word
                    )
                )


        # ================================================================
        # SORT
        # ================================================================

        extracted.sort(
            key=lambda item: (
                item[0],
                -len(item[1])
            )
        )


        # ================================================================
        # DEDUPLICATION
        # ================================================================

        final_aspects = []

        for _, term in extracted:

            clean_term = term.strip()

            if not clean_term:
                continue

            clean_lower = clean_term.lower()

            if (
                clean_lower
                in EXCLUDED_ASPECT_TOKENS
            ):
                continue

            already_covered = False

            for existing in final_aspects:

                existing_lower = existing.lower()

                # Exact duplicate
                if (
                    clean_lower
                    == existing_lower
                ):
                    already_covered = True
                    break

                # If the current term is a single word
                # already contained in a longer aspect phrase,
                # don't add it separately.
                if (
                    len(clean_lower.split()) == 1
                    and clean_lower
                    in existing_lower.split()
                ):
                    already_covered = True
                    break

            if not already_covered:

                final_aspects.append(
                    clean_term
                )


        return final_aspects


# ===========================================================================
# Standalone testing
# ===========================================================================

if __name__ == "__main__":

    extractor = AspectExtractor()

    test_cases = [
        "The food was amazing but the service was extremely slow.",
        "Delicious thin crust pizza and attentive wait staff, though the wine list is small.",
        "Great atmosphere and affordable prices.",
        "The chicken was raw and the soup was freezing cold.",
        "Great beer selection but the atmosphere was far too noisy.",
        "Decent salad, ordinary dressing, but prompt service.",
        "Excellent dessert menu but the prices are exorbitant.",
    ]

    for test_case in test_cases:

        print(
            f"\nReview: '{test_case}'"
        )

        aspects = (
            extractor.extract_aspects(
                test_case
            )
        )

        print(
            "Detected Aspects:",
            aspects
        )