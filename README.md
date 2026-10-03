# Aspect-Based Sentiment Analysis for Restaurant Reviews

An end-to-end, modular, and academically explainable Natural Language Processing (NLP) project built on the **SemEval-2014 Task 4 (Restaurant Reviews)** benchmark dataset.

The system performs **fine-grained aspect-level sentiment classification** using classical NLP and Machine Learning techniques (POS Tagging, Clause Boundary Windowing, TF-IDF, Logistic Regression, and Naive Bayes baseline), accompanied by an interactive **Streamlit** web application and evaluation dashboard.

---

## 📌 Problem Statement & Core Challenge

Standard Sentiment Analysis assigns a single global label to an entire review (e.g., *"Positive"* or *"Negative"*). However, human reviews are inherently multi-faceted:

> *"The food was amazing but the service was extremely slow."*

A traditional classifier is forced to compress this into an ambiguous overall label. In contrast, **Aspect-Based Sentiment Analysis (ABSA)** decomposes the review into target-sentiment pairs:
- **`food`** $\rightarrow$ **Positive** $(75.0\%)$
- **`service`** $\rightarrow$ **Negative** $(84.7\%)$
- **Overall Polarity:** **Mixed (Equally Positive & Negative)**

---

## 📊 Dataset: SemEval-2014 Task 4 (Subtask 1 & 2)

The system is trained and evaluated strictly on the gold-standard benchmark released for SemEval-2014 Task 4:

| Dataset Partition | Total Sentences | Total Aspect Instances | Positive | Negative | Neutral | Conflict* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** (`Restaurants_Train_v2.xml`) | 3,041 | 3,602 | 2,164 | 805 | 633 | Excluded |
| **Gold Test Set** (`Restaurants_Test_Gold.xml`) | 800 | 1,120 | 728 | 196 | 196 | Excluded |

*\*Note: As standard in 3-way ABSA literature, evaluations focus on `positive`, `negative`, and `neutral` polarities.*

---

## 🏗️ Architecture & NLP Pipeline

```
Review Text
    │
    ▼
[Text Preprocessing] ──► Contraction expansion, normalization, tokenization
    │
    ├───────────────────────────────────────────────┐
    ▼                                               ▼
[POS Tagging & Lexicon Filter]             [Clause Boundary Extractor]
    │                                               │
    ▼                                               ▼
Extracted Aspect Terms                     Local Aspect Context Window
(e.g., 'food', 'service')                  (e.g., 'food was amazing', 'service was slow')
    │                                               │
    └───────────────────────┬───────────────────────┘
                            ▼
                [Aspect-Context Pairing]
                            │
                            ▼
          [TF-IDF Vectorizer (Unigrams + Bigrams)]
                            │
                            ▼
          [Logistic Regression (Balanced Weights)]
                 (vs Naive Bayes Baseline)
                            │
                            ▼
          [Aspect-Sentiment Mapping & Aggregation]
                            │
                            ▼
          Streamlit Interactive Dashboard & UI
```

### 1. Preprocessing & Tokenization
- Expands English contractions (`"wasn't"` $\rightarrow$ `"was not"`) to preserve explicit negation signals.
- Preserves punctuation boundaries (`,`, `;`, `.`) necessary for clause segmentation.
- NLTK tokenization.

### 2. POS Tagging & Aspect Term Extraction
- POS Tagging (`nltk.pos_tag`) based on the Penn Treebank tagset.
- Syntactic Chunking: Target entities in restaurants are nouns (`NN`, `NNS`, `NNP`) or compound noun sequences (`<NN|NNS|NNP>+`).
- Sentiment Adjective Stripping: Separates sentiment modifiers (`"delicious"`, `"slow"`) from the target entity (`"food"`, `"service"`).
- Candidate validation using a domain lexicon of 300+ restaurant aspect terms mined from the training corpus.

### 3. Clause Boundary & Context Windowing (Prevents Feature Bleeding)
If global TF-IDF is applied to *"The food was amazing but the service was extremely slow"*, the word *"amazing"* bleeds into the features for *"service"*.
- The pipeline splits sentences along **adversative connectives** (*but, however, yet, although, while, except*) and punctuation.
- For each aspect, it isolates its specific enclosing clause and local $k$-token window ($k=6$).
- Prepend aspect name: `[aspect_term] + [local_context]`.

### 4. Feature Extraction
- **TF-IDF Vectorizer**: Sublinear term frequency scaling, capturing unigrams and bigrams (`ngram_range=(1, 2)`), preserving negations (`"not"`, `"never"`).

### 5. Classification & Baseline
- **Primary Model**: **Logistic Regression** with `class_weight='balanced'` to prevent bias toward the majority positive class.
- **Baseline Model**: **Multinomial Naive Bayes**.

---

## 📈 Model Performance on SemEval Gold Test Set

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **70.27%** | **59.60%** | **61.64%** | **60.16%** | **70.62%** |
| **Naive Bayes (Baseline)** | 67.50% | 70.88% | 39.23% | 37.22% | 57.76% |

### Key Insight for Viva:
Naive Bayes achieves high accuracy on paper solely by classifying nearly everything as `Positive` (Recall = 98.8%, but Negative Recall = only 14.8%). In contrast, **Logistic Regression with balanced class weights** achieves a balanced Macro F1 of **60.16%** with a **65.3% Negative Recall**, correctly distinguishing complaints from praise.

---

## 📂 Project Organization

```
ABSA_Restaurant_Analyzer/
├── data/
│   ├── raw/
│   │   ├── Restaurants_Train_v2.xml      # SemEval-2014 Training XML
│   │   └── Restaurants_Test_Gold.xml     # SemEval-2014 Test XML
│   └── processed/
│       ├── train_aspects.csv             # Processed tabular train data
│       └── test_aspects.csv              # Processed tabular test data
├── models/
│   ├── lr_aspect_classifier.joblib       # Trained Logistic Regression model
│   ├── nb_aspect_classifier.joblib       # Trained Naive Bayes baseline
│   ├── tfidf_vectorizer.joblib           # Fitted TF-IDF N-gram vectorizer
│   ├── aspect_lexicon.joblib             # Mined restaurant domain lexicon
│   └── metrics.json                      # Comprehensive evaluation metrics
├── notebooks/
│   └── exploration_and_evaluation.ipynb  # Interactive EDA & pipeline walkthrough
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # XML parsing & dataset management
│   ├── preprocessor.py                   # Contractions, tokenization, POS tagging
│   ├── aspect_extractor.py               # POS chunking & aspect candidate filtering
│   ├── context_extractor.py              # Clause boundary splitting & windowing
│   ├── classifier.py                     # TF-IDF, model training & evaluation
│   └── pipeline.py                       # Unified end-to-end ABSA pipeline class
├── tests/
│   └── test_reviews.py                   # Verification suite across 12 review cases
├── app.py                                # Streamlit web application & dashboard
├── requirements.txt                      # Project dependencies
└── README.md                             # Academic documentation & viva guide
```

---

## 🚀 Getting Started

### 1. Installation
Install the project requirements:
```bash
pip install -r requirements.txt
```

### 2. Retrain / Evaluate Models (Optional)
To retrain the models from the raw SemEval XML datasets and regenerate artifacts:
```bash
python -m src.classifier
```

### 3. Run Verification Test Suite
Execute the automated test script evaluating 12 real-world cases:
```bash
python tests/test_reviews.py
```

### 4. Launch Streamlit Web Application
Run the interactive UI:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🎯 Verification Test Examples

| ID | Review Text | Detected Aspects | Predicted Polarities | Overall Sentiment |
| :---: | :--- | :--- | :--- | :---: |
| 1 | *"The food was amazing but the service was extremely slow."* | `food`, `service` | `food`: POS (75.0%), `service`: NEG (84.7%) | **Mixed** |
| 2 | *"The pasta was delicious and the ambiance was wonderful."* | `pasta`, `ambiance` | `pasta`: POS (87.1%), `ambiance`: POS (91.8%) | **Positive** |
| 3 | *"The chicken was raw and the soup was freezing cold."* | `chicken`, `soup` | `chicken`: NEG (74.0%), `soup`: NEG (71.9%) | **Negative** |
| 4 | *"The waiters were very attentive and friendly throughout."* | `waiters` | `waiters`: POS (81.8%) | **Positive** |
| 5 | *"Excellent dessert menu but the prices are exorbitant."* | `dessert menu`, `prices`| `dessert menu`: NEU (40.6%), `prices`: NEG (45.4%) | **Negative** |

---

## 🎓 College NLP Viva Q&A Guide

#### Q1: Why not just use BERT or ChatGPT API?
- **Answer:** Black-box LLM APIs incur external latency, financial costs, API token limits, and risk hallucinations. Pre-trained BERT contains hundreds of millions of uninterpretable weights. In an academic viva, classical NLP (TF-IDF + Logistic Regression) demonstrates rigorous mastery of fundamental linguistic representations: POS grammars, n-gram feature extraction, hyperplane separation, and statistical class balancing.

#### Q2: How does the model distinguish between `"not good"` and `"good"`?
- **Answer:** By setting `ngram_range=(1, 2)` in TF-IDF, bigrams such as `"not good"`, `"never again"`, and `"hardly fast"` receive dedicated feature coefficients. Furthermore, preprocessing preserves negation terms instead of purging them as generic stopwords.

#### Q3: Why is class weighting essential in this dataset?
- **Answer:** In restaurant reviews, positive reviews dominate (~58% positive, ~22% negative, ~17% neutral). Unweighted classifiers (like Naive Bayes) bias their priors heavily toward the majority class. Logistic Regression with `class_weight='balanced'` assigns penalty weights inversely proportional to class frequencies ($w_j = \frac{N}{k \cdot n_j}$), ensuring minority complaints receive equal sensitivity.

---

## ⚠️ System Limitations (Academic Integrity)

This project is an academic MVP and not an industrial-scale system:
1. **Implicit Aspects**: Detects explicit noun phrases. Phrases conveying sentiment without naming the aspect (e.g., *"It cost an arm and a leg"*) require semantic frame parsing.
2. **Sarcasm & Irony**: Sarcastic expressions (e.g., *"Only waited 3 hours for a glass of water, superb service!"*) remain challenging for bag-of-words/n-gram features.
3. **Lexicon Coverage**: Aspect candidate identification relies on POS chunking and a 300-term domain lexicon; rare regional dishes may require continual domain lexicon updates.
