"""
classifier.py
=============
Module for training, evaluating, and serializing aspect sentiment classifiers.
- Feature Extraction: TF-IDF with Unigrams and Bigrams
- Main Model: Logistic Regression with balanced class weighting
- Baseline Model: Multinomial Naive Bayes
- Evaluation: Accuracy, Macro F1, Precision, Recall, Confusion Matrix

Academic Explanation:
---------------------
Why Logistic Regression + TF-IDF for ABSA?
1. Interpretability: High feature weights directly correlate with polarity signals
   (e.g., 'not good', 'amazing', 'cold', 'slow'), perfect for college viva defense.
2. Balanced Weighting: In review datasets, positive reviews outnumber negative reviews
   (in SemEval 2014, ~58% positive vs ~22% negative). `class_weight='balanced'`
   adjusts inverse class frequencies to penalize majority bias.
3. N-grams (1, 2): Captures negations ('not clean', 'never again') and intensifiers
   ('extremely good', 'too salty') that unigrams alone miss.
4. Naive Bayes Baseline: Serves as the classic probabilistic benchmark based on
   Bayes' Theorem with conditional word independence assumptions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.data_loader import load_dataset, extract_aspect_lexicon
from src.context_extractor import build_aspect_context_dataset
from src.preprocessor import clean_text


class ABSAClassifier:
    """
    Manages vectorization, training, evaluation, and inference
    for Aspect-Based Sentiment Analysis.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.vectorizer: TfidfVectorizer = None
        self.lr_model: LogisticRegression = None
        self.nb_model: MultinomialNB = None
        self.classes_: list = ["negative", "neutral", "positive"]

    def train_and_evaluate(
        self,
        data_dir: str = "data"
    ) -> Dict[str, Any]:
        """
        Loads the SemEval dataset, constructs aspect-context features,
        trains TF-IDF vectorizer, Logistic Regression, and Naive Bayes models,
        evaluates on test data, and saves all artifacts to disk.

        Returns:
        --------
        Dict[str, Any]:
            Dictionary containing metrics and confusion matrices for both models.
        """
        # Step 1: Load train and test data
        train_df, test_df = load_dataset(data_dir)

        print(f"Building aspect-context features for {len(train_df)} train samples...")
        X_train_raw = build_aspect_context_dataset(
            train_df["text"].tolist(),
            train_df["aspect_term"].tolist()
        )
        y_train = train_df["polarity"].tolist()

        print(f"Building aspect-context features for {len(test_df)} test samples...")
        X_test_raw = build_aspect_context_dataset(
            test_df["text"].tolist(),
            test_df["aspect_term"].tolist()
        )
        y_test = test_df["polarity"].tolist()

        # Step 2: TF-IDF Feature Extraction
        print("Extracting TF-IDF n-gram features (unigrams + bigrams)...")
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            max_features=12000,
            strip_accents="unicode"
        )
        X_train_vec = self.vectorizer.fit_transform(X_train_raw)
        X_test_vec = self.vectorizer.transform(X_test_raw)

        # Step 3: Train Primary Classifier (Logistic Regression)
        print("Training Logistic Regression classifier (balanced class weights)...")
        self.lr_model = LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
        self.lr_model.fit(X_train_vec, y_train)

        # Step 4: Train Baseline Classifier (Multinomial Naive Bayes)
        print("Training Multinomial Naive Bayes baseline...")
        self.nb_model = MultinomialNB(alpha=0.8)
        self.nb_model.fit(X_train_vec, y_train)

        # Step 5: Evaluate Both Models on Gold Test Set
        metrics = self._evaluate_models(X_test_vec, y_test)

        # Step 6: Extract & Save Domain Aspect Lexicon
        lexicon = extract_aspect_lexicon(train_df, min_freq=2)
        lexicon_path = os.path.join(self.models_dir, "aspect_lexicon.joblib")
        joblib.dump(lexicon, lexicon_path)
        print(f"Saved domain aspect lexicon ({len(lexicon)} terms) to {lexicon_path}")

        # Step 7: Serialize Models & Metrics
        self.save_artifacts(metrics)
        return metrics

    def _evaluate_models(self, X_test_vec, y_test) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics for both LR and NB.
        """
        results = {}

        for model_name, model in [("Logistic Regression", self.lr_model), ("Naive Bayes", self.nb_model)]:
            y_pred = model.predict(X_test_vec)
            acc = float(accuracy_score(y_test, y_pred))

            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=0
            )
            prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )

            # Per-class breakdown
            labels = sorted(list(set(y_test)))
            p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
                y_test, y_pred, labels=labels, zero_division=0
            )
            per_class = {
                l: {
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1_score": round(float(f), 4),
                    "support": int(s)
                }
                for l, p, r, f, s in zip(labels, p_cls, r_cls, f_cls, s_cls)
            }

            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

            results[model_name] = {
                "accuracy": round(acc, 4),
                "macro_precision": round(float(prec_macro), 4),
                "macro_recall": round(float(rec_macro), 4),
                "macro_f1": round(float(f1_macro), 4),
                "weighted_f1": round(float(f1_weighted), 4),
                "per_class": per_class,
                "labels": labels,
                "confusion_matrix": cm
            }

        return results

    def save_artifacts(self, metrics: Dict[str, Any]):
        """
        Persists trained models, vectorizer, and metrics JSON to models/ directory.
        """
        joblib.dump(self.vectorizer, os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.lr_model, os.path.join(self.models_dir, "lr_aspect_classifier.joblib"))
        joblib.dump(self.nb_model, os.path.join(self.models_dir, "nb_aspect_classifier.joblib"))

        metrics_file = os.path.join(self.models_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Artifacts successfully saved to {self.models_dir}/")

    def load_artifacts(self):
        """
        Loads pre-trained artifacts from the models directory.
        """
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        lr_path = os.path.join(self.models_dir, "lr_aspect_classifier.joblib")
        nb_path = os.path.join(self.models_dir, "nb_aspect_classifier.joblib")

        if not (os.path.exists(vec_path) and os.path.exists(lr_path)):
            raise FileNotFoundError(
                f"Trained model artifacts not found in {self.models_dir}. "
                "Run classifier.py to train the models first."
            )

        self.vectorizer = joblib.load(vec_path)
        self.lr_model = joblib.load(lr_path)
        if os.path.exists(nb_path):
            self.nb_model = joblib.load(nb_path)

    def predict_aspect_sentiment(
        self,
        aspect_context: str,
        use_model: str = "lr"
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts sentiment for a given aspect-context string.

        Parameters:
        -----------
        aspect_context : str
            Aspect term + local context words.
        use_model : str
            'lr' for Logistic Regression, 'nb' for Naive Bayes.

        Returns:
        --------
        Tuple[str, float, Dict[str, float]]:
            (predicted_polarity, confidence_score, class_probabilities)
        """
        if self.vectorizer is None or self.lr_model is None:
            self.load_artifacts()

        model = self.lr_model if use_model == "lr" else self.nb_model
        vec = self.vectorizer.transform([aspect_context])
        pred_label = model.predict(vec)[0]

        probs = model.predict_proba(vec)[0]
        class_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(model.classes_, probs)}
        confidence = float(max(probs))

        return pred_label, confidence, class_probs


if __name__ == "__main__":
    # Base paths relative to project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data")
    models_path = os.path.join(base_dir, "models")

    classifier = ABSAClassifier(models_dir=models_path)
    eval_metrics = classifier.train_and_evaluate(data_dir=data_path)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS (SemEval-2014 Gold Test Set)")
    print("="*50)
    for model_name, res in eval_metrics.items():
        print(f"\n--- {model_name} ---")
        print(f"Accuracy:        {res['accuracy'] * 100:.2f}%")
        print(f"Macro F1 Score:  {res['macro_f1'] * 100:.2f}%")
        print(f"Weighted F1:     {res['weighted_f1'] * 100:.2f}%")
        print("Per-class performance:")
        for cls_name, p_res in res["per_class"].items():
            print(f"  {cls_name.capitalize():<8}: Precision={p_res['precision']:.3f}, Recall={p_res['recall']:.3f}, F1={p_res['f1_score']:.3f} (N={p_res['support']})")
