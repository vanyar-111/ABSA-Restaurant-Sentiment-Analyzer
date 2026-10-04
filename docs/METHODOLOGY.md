# Methodology

Detailed description of the pipeline used in the Aspect-Based Sentiment Analysis system.

## 1. Text Preprocessing

- Expands English contractions so that explicit negation signals are preserved.
- Preserves punctuation boundaries required for clause segmentation.
- Tokenizes with NLTK.
- Normalizes text before feature extraction.

## 2. Aspect Term Extraction

Aspects are extracted using linguistic and domain-specific information:

- POS tagging with the Penn Treebank tagset.
- Noun and compound-noun patterns.
- A restaurant-domain aspect vocabulary.
- Filtering of sentiment modifiers from candidate aspect terms.

Examples:

```text
food
service
pizza
beer selection
dessert menu
prices
atmosphere
```

For benchmark evaluation, the gold aspect terms supplied by SemEval are used to build the training and test aspect-context pairs.

For new user reviews, candidate aspects come from the POS-based and domain-lexicon approach above. As a result, benchmark scores measure sentiment classification, not extraction quality.

## 3. Local Context Extraction

Using the whole sentence as context lets sentiment about one aspect influence the prediction for another.

For example:

"The food was amazing but the service was extremely slow."

The sentence-level context for service contains the strongly positive word amazing.

To reduce this overlap, the pipeline:

Identifies clause boundaries.
Splits around adversative conjunctions such as but, however, although, yet, and while.
Selects the clause containing the target aspect.
Extracts a local context window around the aspect.
Builds the aspect-context representation used for classification.

Result:

```text
food    -> food was amazing
service -> service was extremely slow
```

## 4. TF-IDF Features

The aspect-context representations are converted into numerical features using TF-IDF.

The current vectorizer configuration is:

```python
TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=1,
    sublinear_tf=True,
    max_features=12000,
    strip_accents="unicode",
)
``` 

The vectorizer captures:

- Unigrams
- Bigrams
- Trigrams

This allows the model to capture local expressions such as:

- not good
- never again
- extremely good
- too salty

rather than relying only on individual words.

## 5. Classifiers

```python
Logistic Regression (Primary Model)
LogisticRegression(
    C=2.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
```

Balanced class weighting is used because the training set contains far more positive examples than negative or neutral ones. This helps reduce the tendency of the classifier to favor the majority class.

```python
Multinomial Naive Bayes (Baseline)
MultinomialNB(alpha=0.8)
```

Multinomial Naive Bayes provides a classical probabilistic baseline for comparison against Logistic Regression.

## 6. Evaluation Metrics

The system reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix

Macro-averaged metrics give each sentiment class equal weight, which makes them more informative than accuracy on an imbalanced dataset.

Model selection is based on macro metrics because Naive Bayes reaches higher macro precision but much lower macro recall and macro F1, with predictions concentrated toward the positive class.

## 7. Overall Review Polarity

Aspect-level predictions are aggregated into one review-level label:

- Positive
- Negative
- Neutral
- Mixed

A review containing both positive and negative aspect predictions can therefore receive an overall Mixed polarity.

For example:

```text
food    -> Positive
service -> Negative

Overall -> Mixed
``` 

## 8. Pipeline Summary

```text
Review Text
    |
    v
Text Preprocessing
    |
    v
POS Tagging & Aspect Extraction
    |
    v
Local Context Extraction
    |
    v
Aspect-Context Pairing
    |
    v
TF-IDF Features (1-3 grams)
    |
    +-----------------------------+
    |                             |
    v                             v
Logistic Regression       Multinomial Naive Bayes
    |                             |
    +-------------+---------------+
                  |
                  v
        Aspect-Level Sentiment
                  |
                  v
        Overall Review Polarity
```