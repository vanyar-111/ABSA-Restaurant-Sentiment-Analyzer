"""
pipeline.py
===========
Unified End-to-End Aspect-Based Sentiment Analysis (ABSA) Pipeline.

Academic Explanation:
---------------------
This module links together each individual stage of the classic NLP pipeline:
1. Input Review -> 2. Text Preprocessing -> 3. POS Aspect Extraction
-> 4. Aspect-Context Isolation -> 5. TF-IDF Representation
-> 6. Classification -> 7. Aspect-Sentiment Mapping
-> 8. Overall Review Sentiment Aggregation

Example Execution:
------------------
Input:
  "The food was amazing but the service was extremely slow."

Pipeline Execution:
  1. Preprocessing: expand contractions, clean whitespace.
  2. POS Aspect Extraction: extracts ['food', 'service'].
  3. Context Extraction:
     - 'food' -> 'food The food was amazing'
     - 'service' -> 'service the service was extremely slow'
  4. TF-IDF + Logistic Regression:
     - 'food' -> positive (e.g. 0.94 confidence)
     - 'service' -> negative (e.g. 0.91 confidence)
  5. Aggregation: Mixed sentiment (1 Positive, 1 Negative).
"""

import os
import joblib
from typing import Dict, List, Any, Optional

from src.preprocessor import clean_text
from src.aspect_extractor import AspectExtractor
from src.context_extractor import extract_aspect_context
from src.classifier import ABSAClassifier


class ABSAPipeline:
    """
    Unified end-to-end interface for Aspect-Based Sentiment Analysis.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "models")
            )
        self.models_dir = models_dir

        # Initialize Classifier
        self.classifier = ABSAClassifier(models_dir=self.models_dir)
        try:
            self.classifier.load_artifacts()
            lexicon_path = os.path.join(self.models_dir, "aspect_lexicon.joblib")
            if os.path.exists(lexicon_path):
                domain_lex = joblib.load(lexicon_path)
            else:
                domain_lex = None
        except Exception:
            # If not yet trained, extractor will use fallback seed lexicon
            domain_lex = None

        # Initialize Aspect Extractor
        self.aspect_extractor = AspectExtractor(domain_lexicon=domain_lex)

    def analyze_review(
        self,
        review_text: str,
        classifier_type: str = "lr",
        target_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a single restaurant review end-to-end.

        Parameters:
        -----------
        review_text : str
            Raw restaurant review text.
        classifier_type : str
            'lr' (Logistic Regression, default) or 'nb' (Naive Bayes).
        target_aspects : Optional[List[str]]
            Specific aspects to evaluate if user provides them explicitly.
            If None, aspects are automatically extracted using POS + Lexicon.

        Returns:
        --------
        Dict[str, Any]:
            Dictionary containing:
            - 'review': Original review text
            - 'cleaned_review': Preprocessed text
            - 'detected_aspects': List of detected aspect strings
            - 'aspect_results': List of dicts per aspect:
                - 'aspect': aspect term
                - 'sentiment': 'positive', 'negative', or 'neutral'
                - 'confidence': confidence score (0.0 - 1.0)
                - 'probabilities': dict of class probabilities
                - 'context_used': isolated local context text
            - 'overall_sentiment': Overall review polarity assessment
            - 'sentiment_summary': Counts of positive, negative, neutral aspects
        """
        cleaned = clean_text(review_text)

        # Step 1: Aspect Term Extraction
        if target_aspects is not None and len(target_aspects) > 0:
            aspects = target_aspects
        else:
            aspects = self.aspect_extractor.extract_aspects(cleaned)

        aspect_results = []
        counts = {"positive": 0, "negative": 0, "neutral": 0}

        # Step 2: Context Extraction & Sentiment Classification per Aspect
        for aspect in aspects:
            # Extract localized context
            aspect_ctx = extract_aspect_context(cleaned, aspect)

            # Classify sentiment
            polarity, conf, probs = self.classifier.predict_aspect_sentiment(
                aspect_ctx, use_model=classifier_type
            )

            counts[polarity] += 1
            aspect_results.append({
                "aspect": aspect,
                "sentiment": polarity,
                "confidence": round(conf, 4),
                "probabilities": probs,
                "context_used": aspect_ctx
            })

        # Step 3: Compute Overall Sentiment
        overall_sentiment = self._aggregate_overall_sentiment(
            cleaned, aspect_results, counts, classifier_type
        )

        return {
            "review": review_text,
            "cleaned_review": cleaned,
            "detected_aspects": aspects,
            "aspect_results": aspect_results,
            "overall_sentiment": overall_sentiment,
            "sentiment_summary": counts
        }

    def _aggregate_overall_sentiment(
        self,
        cleaned_text: str,
        aspect_results: List[Dict[str, Any]],
        counts: Dict[str, int],
        classifier_type: str
    ) -> str:
        """
        Synthesizes aspect sentiments into an overall review assessment.
        Handles mixed sentiments with nuance.
        """
        if not aspect_results:
            # Fallback: classify entire sentence as general sentiment
            try:
                polarity, _, _ = self.classifier.predict_aspect_sentiment(
                    f"general {cleaned_text}", use_model=classifier_type
                )
                return f"{polarity.capitalize()} (Overall sentence)"
            except Exception:
                return "Neutral (No aspects detected)"

        pos = counts["positive"]
        neg = counts["negative"]
        neu = counts["neutral"]

        # Mixed sentiment detection
        if pos > 0 and neg > 0:
            if pos == neg:
                return "Mixed (Equally Positive & Negative)"
            elif pos > neg:
                return "Mixed (Predominantly Positive)"
            else:
                return "Mixed (Predominantly Negative)"
        elif pos > 0 and neg == 0:
            return "Positive"
        elif neg > 0 and pos == 0:
            return "Negative"
        else:
            return "Neutral"


if __name__ == "__main__":
    pipeline = ABSAPipeline()
    sample = "The food was amazing but the service was extremely slow."
    res = pipeline.analyze_review(sample)
    print("\nReview Analysis Result:")
    print("Review:", res["review"])
    print("Overall Sentiment:", res["overall_sentiment"])
    print("Aspects:")
    for a in res["aspect_results"]:
        print(f"  - {a['aspect']}: {a['sentiment'].upper()} (conf: {a['confidence']}) | Context: '{a['context_used']}'")
