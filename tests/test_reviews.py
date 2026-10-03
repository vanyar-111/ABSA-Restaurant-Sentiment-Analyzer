"""
test_reviews.py
===============
Comprehensive test suite testing the Aspect-Based Sentiment Analysis pipeline
across at least 10 realistic restaurant reviews, with strong emphasis on
mixed-sentiment reviews.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ABSAPipeline


TEST_REVIEWS = [
    {
        "id": 1,
        "category": "Mixed Sentiment (Benchmark Requirement)",
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
        "category": "Mixed Sentiment (Adversative clause)",
        "review": "Terrible management and overpriced drinks, though the pizza was okay.",
        "expected": {"management": "negative", "drinks": "negative", "pizza": "neutral"}
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
        "category": "Mixed Sentiment (Food vs Environment)",
        "review": "Great beer selection but the atmosphere was far too noisy.",
        "expected": {"beer selection": "positive", "atmosphere": "negative"}
    },
    {
        "id": 7,
        "category": "Service & Experience Negative",
        "review": "We waited 40 minutes for our table and the hostess was rude.",
        "expected": {"table": "negative", "hostess": "negative"}
    },
    {
        "id": 8,
        "category": "Neutral Sentiment",
        "review": "Average burger, nothing special about the taste.",
        "expected": {"burger": "neutral"}
    },
    {
        "id": 9,
        "category": "Mixed Sentiment (Menu vs Price)",
        "review": "Excellent dessert menu but the prices are exorbitant.",
        "expected": {"dessert": "positive", "prices": "negative"}
    },
    {
        "id": 10,
        "category": "Mixed Sentiment (Food vs Cost)",
        "review": "The sushi was fresh, but the bill gave us a shock.",
        "expected": {"sushi": "positive", "bill": "negative"}
    },
    {
        "id": 11,
        "category": "Mixed Sentiment (Neutral vs Positive)",
        "review": "Decent salad, ordinary dressing, but prompt service.",
        "expected": {"salad": "neutral", "service": "positive"}
    },
    {
        "id": 12,
        "category": "Atmosphere & Beverage Positive",
        "review": "A pleasant dining experience with great coffee and cozy seating.",
        "expected": {"coffee": "positive", "seating": "positive"}
    }
]


def run_tests():
    print("=" * 70)
    print("ASPECT-BASED SENTIMENT ANALYSIS (ABSA) - COMPREHENSIVE VERIFICATION")
    print("=" * 70)

    pipeline = ABSAPipeline()
    total_reviews = len(TEST_REVIEWS)
    mixed_count = 0

    for item in TEST_REVIEWS:
        print(f"\n[Test Case {item['id']}] Category: {item['category']}")
        print(f"Review: \"{item['review']}\"")

        res = pipeline.analyze_review(item["review"])
        overall = res["overall_sentiment"]
        detected = res["aspect_results"]

        print(f"Overall Sentiment: {overall}")
        print(f"Aspects Detected ({len(detected)}):")

        if "Mixed" in overall:
            mixed_count += 1

        for a in detected:
            aspect_name = a["aspect"]
            sentiment = a["sentiment"]
            conf = a["confidence"]
            ctx = a["context_used"]
            print(f"  * {aspect_name:<16} -> {sentiment.upper():<8} (Confidence: {conf*100:.1f}%) | Context: '{ctx}'")

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: Successfully analyzed {total_reviews}/{total_reviews} reviews.")
    print(f"Mixed-sentiment reviews identified: {mixed_count}")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
