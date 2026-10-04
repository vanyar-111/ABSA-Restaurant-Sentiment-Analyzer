
````markdown
# Aspect-Based Sentiment Analysis for Restaurant Reviews

An end-to-end Natural Language Processing system for fine-grained sentiment analysis of restaurant reviews using the SemEval-2014 Task 4 Restaurant Reviews benchmark dataset.

The system identifies individual aspects within a review, extracts their local context, and predicts sentiment for each aspect using classical NLP and machine learning techniques. It includes POS-based aspect extraction, clause-aware context extraction, TF-IDF feature representation, Logistic Regression classification, a Multinomial Naive Bayes baseline, and an interactive Streamlit application.

---

## Overview

Traditional sentiment analysis assigns a single sentiment label to an entire review. This can be insufficient when a review contains different opinions about different aspects.

For example:

> "The food was amazing but the service was extremely slow."

A conventional sentiment classifier may assign one overall label to the review. Aspect-Based Sentiment Analysis instead identifies the individual targets and their associated sentiments:

| Aspect | Sentiment |
|---|---|
| food | Positive |
| service | Negative |

The system provides both aspect-level predictions and an aggregated overall polarity for the review.

---

## Key Results

Evaluation is performed on the SemEval-2014 gold test set using the three sentiment classes: positive, negative, and neutral.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **70.54%** | **60.31%** | **62.15%** | **60.86%** | **70.79%** |
| Multinomial Naive Bayes | 66.96% | 77.53% | 37.70% | 34.62% | 56.14% |

**Primary model:** Logistic Regression with balanced class weights

**Features:** TF-IDF unigrams, bigrams, and trigrams

**Dataset:** SemEval-2014 Task 4 Restaurant Reviews

**Test set:** 1,120 aspect instances after excluding conflict-labelled instances

The Logistic Regression model is selected as the primary model because it provides substantially better balance across the three sentiment classes. Naive Bayes achieves higher macro precision, but its macro recall and macro F1 are considerably lower because its predictions are strongly concentrated toward the majority positive class.

---

## Dataset

The system uses the **SemEval-2014 Task 4 Restaurant Reviews** dataset.

The dataset contains restaurant reviews annotated with explicit aspect terms and their associated sentiment polarity.

| Dataset Partition | Sentences | Aspect Instances | Positive | Negative | Neutral |
|---|---:|---:|---:|---:|---:|
| Training Set | 3,041 | 3,602 | 2,164 | 805 | 633 |
| Gold Test Set | 800 | 1,120 | 728 | 196 | 196 |

Conflict-labelled instances are excluded from the three-class sentiment classification task.

The training data is used to construct aspect-context features and train the sentiment classifiers. The gold test set is used for quantitative evaluation.

### Evaluation Scope

The benchmark evaluation uses the gold aspect terms supplied by SemEval when constructing the aspect-context pairs.

Therefore, the reported benchmark scores measure **aspect-level sentiment classification**, not end-to-end aspect extraction and sentiment classification.

For new user reviews, candidate aspects are extracted using the implemented POS-based and restaurant-domain lexicon approach.

---

## NLP Pipeline

```text
                         Review Text
                              |
                              v
                    Text Preprocessing
                              |
                              v
                 POS Tagging & Tokenization
                              |
                              v
                    Aspect Extraction
                              |
                              v
                 Local Context Extraction
                              |
                              v
                  Aspect-Context Pairing
                              |
                              v
                   TF-IDF Representation
                         (1-3 grams)
                              |
                 +------------+------------+
                 |                         |
                 v                         v
       Logistic Regression        Multinomial Naive Bayes
          Primary Model                Baseline
                 |                         |
                 +------------+------------+
                              |
                              v
                  Aspect-Level Sentiment
                              |
                              v
                   Overall Review Polarity
                              |
                              v
                    Streamlit Application
````

Detailed descriptions of the pipeline, feature extraction, classifiers, and evaluation methodology are available in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## How It Works

### 1. Text Preprocessing

The preprocessing stage prepares review text for feature extraction.

The pipeline:

* Expands English contractions to preserve explicit negation signals.
* Preserves punctuation boundaries required for clause segmentation.
* Tokenizes review text using NLTK.
* Normalizes text before feature extraction.

---

### 2. Aspect Term Extraction

Restaurant aspects are extracted using linguistic and domain-specific information.

The extraction process uses:

* POS tagging with the Penn Treebank tagset.
* Noun and compound-noun patterns.
* Restaurant-domain aspect vocabulary.
* Filtering of sentiment modifiers from candidate aspect terms.

Examples of extracted aspects include:

```text
food
service
pizza
beer selection
dessert menu
prices
atmosphere
```

For benchmark evaluation, the gold aspect terms supplied by SemEval are used to construct the training and test aspect-context pairs.

For new reviews, candidate aspects are extracted using the implemented POS-based and domain-lexicon approach.

---

### 3. Local Context Extraction

A key design choice is the use of aspect-specific local context.

Consider:

> "The food was amazing but the service was extremely slow."

If the entire sentence is used as context for both aspects, the strongly positive word `amazing` may influence the sentiment prediction for `service`.

To reduce this overlap, the pipeline:

1. Identifies clause boundaries.
2. Splits around adversative conjunctions such as `but`, `however`, `although`, `yet`, and `while`.
3. Selects the clause containing the target aspect.
4. Extracts a local context window around the aspect.
5. Builds the aspect-context representation used for classification.

For the example above:

```text
food    -> food was amazing
service -> service was extremely slow
```

This provides the classifier with more aspect-specific sentiment context.

---

### 4. TF-IDF Feature Extraction

The aspect-context representations are converted into numerical features using TF-IDF.

The current vectorizer configuration is:

```python
TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=1,
    sublinear_tf=True,
    max_features=12000,
    strip_accents="unicode"
)
```

The vectorizer captures:

* Unigrams
* Bigrams
* Trigrams

N-gram features allow the model to capture local expressions such as:

```text
not good
never again
extremely good
too salty
```

rather than relying only on individual words.

---

## Classification Models

### Logistic Regression

Logistic Regression is the primary sentiment classifier.

The current configuration is:

```python
LogisticRegression(
    C=2.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
```

Balanced class weighting is used because the dataset contains substantially more positive examples than negative and neutral examples.

This reduces the tendency of the classifier to favor the majority class.

### Multinomial Naive Bayes

Multinomial Naive Bayes is included as a classical probabilistic baseline.

```python
MultinomialNB(alpha=0.8)
```

It provides a comparison against Logistic Regression using the same TF-IDF feature representation.

---

## Model Performance

Evaluation is performed on the SemEval-2014 gold test set.

| Model                   |   Accuracy | Macro Precision | Macro Recall |   Macro F1 | Weighted F1 |
| ----------------------- | ---------: | --------------: | -----------: | ---------: | ----------: |
| Logistic Regression     | **70.54%** |      **60.31%** |   **62.15%** | **60.86%** |  **70.79%** |
| Multinomial Naive Bayes |     66.96% |          77.53% |       37.70% |     34.62% |      56.14% |

### Why Logistic Regression?

Naive Bayes achieves higher macro precision but substantially lower macro recall and macro F1.

Its predictions are strongly concentrated toward the majority positive class. Logistic Regression with balanced class weights provides a better balance between positive, negative, and neutral predictions.

For this reason, Logistic Regression is used as the primary model in the application.

---

## Evaluation Metrics

The system reports:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Weighted F1
* Per-class Precision
* Per-class Recall
* Per-class F1
* Confusion Matrix

Macro-averaged metrics give each sentiment class equal importance and are therefore useful when evaluating an imbalanced dataset.

---

## Overall Review Polarity

After aspect-level sentiment predictions are generated, they are aggregated into a review-level polarity.

The application can report:

* Positive
* Negative
* Neutral
* Mixed

A review containing both positive and negative aspect predictions can therefore be represented as **Mixed** even when no single sentiment label accurately describes the entire review.

For example:

```text
The food was amazing but the service was extremely slow.

food     -> Positive
service  -> Negative

Overall  -> Mixed
```

---

## Streamlit Application

The project includes an interactive Streamlit application for analyzing restaurant reviews.

### Review Analyzer

Users can enter a restaurant review and receive:

* Detected aspects
* Aspect-level sentiment
* Prediction confidence
* Local context used for each aspect
* Overall review polarity

Example:

```text
Input:
The food was amazing but the service was extremely slow.

Output:

FOOD
Sentiment: Positive

SERVICE
Sentiment: Negative

Overall:
Mixed
```

### Model and Dataset Insights

This page provides information about:

* Dataset composition
* Sentiment distribution
* Aspect statistics
* NLP pipeline
* Model configuration

### Model Evaluation

This page displays:

* Accuracy
* Precision
* Recall
* F1 scores
* Class-wise performance
* Confusion matrices
* Logistic Regression vs. Naive Bayes comparison

### Project Documentation

This page provides an overview of:

* System architecture
* Methodology
* Dataset
* Models
* Evaluation

---

## Project Structure

```text
ABSA_Restaurant_Analyzer/
│
├── data/
│   ├── raw/
│   │   ├── Restaurants_Train_v2.xml
│   │   └── Restaurants_Test_Gold.xml
│   │
│   └── processed/
│       ├── train_aspects.csv
│       └── test_aspects.csv
│
├── docs/
│   └── METHODOLOGY.md
│
├── models/
│   ├── lr_aspect_classifier.joblib
│   ├── nb_aspect_classifier.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── aspect_lexicon.joblib
│   └── metrics.json
│
├── notebooks/
│   └── exploration_and_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── aspect_extractor.py
│   ├── context_extractor.py
│   ├── classifier.py
│   └── pipeline.py
│
├── tests/
│   └── test_reviews.py
│
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

### Requirements

* Python 3.9 or later
* pip

### Clone the Repository

```bash
git clone https://github.com/vanyar-111/ABSA-Restaurant-Sentiment-Analyzer.git
cd ABSA-Restaurant-Sentiment-Analyzer
```

### Create a Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The serialized models were created using **scikit-learn 1.6.1**, which is pinned in `requirements.txt` for compatibility with the saved model artifacts.

---

## NLTK Data

The application requires the NLTK resources used by the preprocessing and POS-tagging pipeline.

Run the following once:

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']]"
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

Open the address in a browser to access the application.

---

## Retraining the Models

The models can be retrained from the SemEval XML datasets.

The required files should be placed in:

```text
data/raw/
├── Restaurants_Train_v2.xml
└── Restaurants_Test_Gold.xml
```

Run:

```bash
python -m src.classifier
```

The training process:

1. Loads the training and test datasets.
2. Builds aspect-context representations.
3. Fits the TF-IDF vectorizer.
4. Trains Logistic Regression.
5. Trains Multinomial Naive Bayes.
6. Evaluates both models.
7. Generates evaluation metrics.
8. Saves the trained model artifacts.
9. Saves `metrics.json`.

The generated artifacts are stored in the `models/` directory.

---

## Verification

The project contains a verification suite covering a range of restaurant review examples.

Run:

```bash
python tests/test_reviews.py
```

The verification cases include:

* Positive reviews
* Negative reviews
* Mixed-sentiment reviews
* Multiple aspects
* Aspect-specific sentiment
* Local context separation

The verification suite is intended as an application-level sanity check.

The primary quantitative benchmark remains the SemEval gold test-set evaluation described in the Model Performance section.

---

## Limitations

### Implicit Aspects

The aspect extraction system primarily targets explicit noun-based aspects. Reviews expressing sentiment without explicitly naming an aspect may not be identified correctly.

### Sarcasm and Irony

Bag-of-words and n-gram representations have limited ability to understand sarcasm, irony, and context-dependent humour.

### Domain Vocabulary

Aspect extraction depends partly on restaurant-domain vocabulary. Rare dishes, regional terminology, or previously unseen expressions may require additional vocabulary coverage.

### Complex Context

Long-distance dependencies, multiple clauses, and complex linguistic constructions can reduce the reliability of local context extraction.

### Class Imbalance

The benchmark contains substantially more positive examples than negative and neutral examples. Balanced class weighting reduces this effect, but minority-class prediction remains challenging.

### Evaluation Scope

The reported benchmark scores use gold aspect terms and therefore evaluate sentiment classification rather than complete end-to-end ABSA performance.

---

## Future Improvements

Potential extensions include:

* Transformer-based contextual representations.
* BERT-based aspect extraction and sentiment classification.
* Joint aspect extraction and sentiment classification.
* Dependency parsing for improved aspect-opinion association.
* Improved handling of implicit aspects.
* More advanced negation and sarcasm detection.
* Aspect-opinion pair extraction.
* Domain adaptation for other review domains.
* End-to-end evaluation using predicted aspect terms.

---

## Technologies

### Programming and Application

* Python
* Streamlit
* Joblib

### NLP

* NLTK
* Text preprocessing
* Tokenization
* POS tagging
* Noun phrase extraction
* Domain lexicon filtering
* Clause boundary detection
* Context windowing
* TF-IDF
* N-gram features

### Machine Learning

* Scikit-learn
* Logistic Regression
* Multinomial Naive Bayes
* Class-weight balancing
* Confusion matrix evaluation
* Precision, Recall, and F1 evaluation

### Data and Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## License

The source code in this repository is released under the MIT License. See [`LICENSE`](LICENSE).

The SemEval-2014 Task 4 dataset remains subject to its own terms of use.
