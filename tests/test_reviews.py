"""
test_reviews.py
===============
Evaluation tests for the Aspect-Based Sentiment Analysis pipeline.

The test suite checks:
1. Whether expected aspects are detected.
2. Whether their predicted sentiment matches the expected sentiment.
3. Aspect-level precision, recall and F1.
4. Review-level exact-match accuracy.
5. Mixed-sentiment review handling.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.pipeline import ABSAPipeline


TEST_REVIEWS = [
    {
        "id": 1,
        "category": "Mixed Sentiment",
        "review": "The food was amazing but the service was extremely slow.",
        "expected": {"food": "positive", "service": "negative"}
    },
    {
        "id": 2,
        "category": "Multi-Aspect Positive",
        "review": "The pasta was delicious and the ambiance was wonderful.",
        "expected": {"pasta": "positive", "ambiance": "positive"}
    },
    {
        "id": 3,
        "category": "Mixed Sentiment",
        "review": "Terrible management and overpriced drinks, though the pizza was okay.",
        "expected": {
            "management": "negative",
            "drinks": "negative",
            "pizza": "neutral"
        }
    },
    {
        "id": 4,
        "category": "Service Positive",
        "review": "The waiters were very attentive and friendly throughout the evening.",
        "expected": {"waiters": "positive"}
    },
    {
        "id": 5,
        "category": "Food Multi-Negative",
        "review": "The chicken was raw and the soup was freezing cold.",
        "expected": {"chicken": "negative", "soup": "negative"}
    },
    {
        "id": 6,
        "category": "Mixed Sentiment",
        "review": "Great beer selection but the atmosphere was far too noisy.",
        "expected": {
            "beer selection": "positive",
            "atmosphere": "negative"
        }
    },
    {
        "id": 7,
        "category": "Service & Experience Negative",
        "review": "We waited 40 minutes for our table and the hostess was rude.",
        "expected": {
            "table": "negative",
            "hostess": "negative"
        }
    },
    {
        "id": 8,
        "category": "Neutral Sentiment",
        "review": "Average burger, nothing special about the taste.",
        "expected": {"burger": "neutral"}
    },
    {
        "id": 9,
        "category": "Mixed Sentiment",
        "review": "Excellent dessert menu but the prices are exorbitant.",
        "expected": {
            "dessert menu": "positive",
            "prices": "negative"
        }
    },
    {
        "id": 10,
        "category": "Mixed Sentiment",
        "review": "The sushi was fresh, but the bill gave us a shock.",
        "expected": {
            "sushi": "positive",
            "bill": "negative"
        }
    },
    {
        "id": 11,
        "category": "Mixed Sentiment",
        "review": "Decent salad, ordinary dressing, but prompt service.",
        "expected": {
            "salad": "neutral",
            "service": "positive"
        }
    },
    {
        "id": 12,
        "category": "Atmosphere & Beverage Positive",
        "review": "A pleasant dining experience with great coffee and cozy seating.",
        "expected": {
            "coffee": "positive",
            "seating": "positive"
        }
    }
]


def normalize_aspect(aspect):
    """Normalize aspect names for comparison."""
    return " ".join(aspect.lower().strip().split())


def get_predictions(aspect_results):
    """
    Convert pipeline output into:
        {aspect_name: predicted_sentiment}
    """
    predictions = {}

    for result in aspect_results:
        aspect = normalize_aspect(result["aspect"])
        sentiment = result["sentiment"].lower().strip()

        predictions[aspect] = sentiment

    return predictions


def calculate_metrics(total_tp, total_fp, total_fn):
    """Calculate precision, recall and F1."""
    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1


def run_tests():
    print("=" * 80)
    print("ABSA PIPELINE - REVIEW LEVEL EVALUATION")
    print("=" * 80)

    pipeline = ABSAPipeline()

    total_expected_aspects = 0
    total_predicted_aspects = 0
    total_correct = 0

    review_exact_matches = 0
    mixed_reviews = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for item in TEST_REVIEWS:

        print(f"\n[Test Case {item['id']}] {item['category']}")
        print(f"Review: \"{item['review']}\"")

        result = pipeline.analyze_review(item["review"])

        predicted = get_predictions(result["aspect_results"])

        expected = {
            normalize_aspect(aspect): sentiment.lower()
            for aspect, sentiment in item["expected"].items()
        }

        total_expected_aspects += len(expected)
        total_predicted_aspects += len(predicted)

        # Compare every expected aspect
        case_correct = 0

        for aspect, expected_sentiment in expected.items():

            if aspect in predicted:

                if predicted[aspect] == expected_sentiment:
                    true_positive += 1
                    total_correct += 1
                    case_correct += 1

                    print(
                        f"  PASS  {aspect}: "
                        f"{predicted[aspect]}"
                    )

                else:
                    false_negative += 1
                    false_positive += 1

                    print(
                        f"  WRONG {aspect}: "
                        f"expected={expected_sentiment}, "
                        f"predicted={predicted[aspect]}"
                    )

            else:
                false_negative += 1

                print(
                    f"  MISSED {aspect}: "
                    f"expected={expected_sentiment}"
                )

        # Anything predicted that wasn't expected is a false positive
        unexpected = set(predicted) - set(expected)

        false_positive += len(unexpected)

        for aspect in unexpected:
            print(
                f"  EXTRA  {aspect}: "
                f"predicted={predicted[aspect]}"
            )

        # Exact match means every expected aspect/sentiment pair
        # matches and there are no extra predicted aspects.
        if predicted == expected:
            review_exact_matches += 1

        # Check whether the expected review contains mixed sentiment.
        expected_sentiments = set(expected.values())

        if len(expected_sentiments) > 1:
            mixed_reviews += 1

        print(
            f"  Review result: "
            f"{case_correct}/{len(expected)} expected aspects correct"
        )

        print(
            f"  Overall sentiment: "
            f"{result['overall_sentiment']}"
        )

    # Metrics
    precision, recall, f1 = calculate_metrics(
        true_positive,
        false_positive,
        false_negative
    )

    aspect_accuracy = (
        total_correct / total_expected_aspects
        if total_expected_aspects > 0
        else 0
    )

    review_accuracy = (
        review_exact_matches / len(TEST_REVIEWS)
        if TEST_REVIEWS
        else 0
    )

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    print(
        f"Reviews evaluated:           {len(TEST_REVIEWS)}"
    )

    print(
        f"Expected aspect instances:   {total_expected_aspects}"
    )

    print(
        f"Predicted aspect instances:  {total_predicted_aspects}"
    )

    print(
        f"Correct aspect-sentiments:    {total_correct}"
    )

    print(
        f"Aspect-level accuracy:        {aspect_accuracy * 100:.2f}%"
    )

    print(
        f"Aspect-level precision:       {precision * 100:.2f}%"
    )

    print(
        f"Aspect-level recall:          {recall * 100:.2f}%"
    )

    print(
        f"Aspect-level F1:              {f1 * 100:.2f}%"
    )

    print(
        f"Review exact-match accuracy:  {review_accuracy * 100:.2f}%"
    )

    print(
        f"Mixed-sentiment cases:        {mixed_reviews}"
    )

    print("=" * 80)

    return {
        "aspect_accuracy": aspect_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "review_accuracy": review_accuracy,
    }


if __name__ == "__main__":
    run_tests()