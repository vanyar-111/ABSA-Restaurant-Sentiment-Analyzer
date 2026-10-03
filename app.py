"""
app.py
======
Interactive Streamlit Web Application & Dashboard for:
"Aspect-Based Sentiment Analysis for Restaurant Reviews"

College NLP Project MVP
- Trained on SemEval-2014 Task 4 Restaurant Reviews dataset
- Classical NLP: NLTK POS Tagging, TF-IDF N-grams, Logistic Regression & Naive Bayes
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis | Restaurant Reviews",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure src package can be imported
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.pipeline import ABSAPipeline
from src.data_loader import load_dataset


# Cache pipeline and dataset loader to ensure instantaneous responsiveness
@st.cache_resource
def get_pipeline():
    models_path = os.path.join(APP_DIR, "models")
    return ABSAPipeline(models_dir=models_path)


@st.cache_data
def get_dataset_stats():
    data_path = os.path.join(APP_DIR, "data")
    train_df, test_df = load_dataset(data_path)
    return train_df, test_df


@st.cache_data
def get_model_metrics():
    metrics_path = os.path.join(APP_DIR, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


pipeline = get_pipeline()
train_df, test_df = get_dataset_stats()
metrics_data = get_model_metrics()


# Styling CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .sentiment-positive {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .sentiment-negative {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .sentiment-neutral {
        background-color: #E5E7EB;
        color: #374151;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .sentiment-mixed {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .aspect-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Navigation & Settings
st.sidebar.title("🍽️ ABSA Restaurant NLP")
st.sidebar.markdown("**SemEval-2014 Task 4 Project**")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Live Review Analyzer", "📊 Dataset & Model Dashboard", "🧪 Batch Benchmark Tester", "📖 Viva & Architecture Q&A"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Classifier Engine")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression (Balanced)", "Naive Bayes (Baseline)"],
    index=0
)
model_code = "lr" if "Logistic" in model_choice else "nb"

st.sidebar.markdown("---")
st.sidebar.info(
    "**Academically Understandable MVP**\n\n"
    "• Dataset: SemEval-2014 Task 4\n\n"
    "• Feature: TF-IDF (1-2 N-grams)\n\n"
    "• Context: Clause boundary windowing\n\n"
    "• No LLM/OpenAI APIs utilized."
)


# ==========================================
# PAGE 1: LIVE REVIEW ANALYZER
# ==========================================
if page == "🔍 Live Review Analyzer":
    st.markdown('<div class="main-title">Aspect-Based Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Extract fine-grained restaurant aspects and determine their individual sentiment polarities.</div>', unsafe_allow_html=True)

    # Example Review Quick-Selector
    examples = {
        "Custom Review": "",
        "Mixed: Food vs Service (Prompt Requirement)": "The food was amazing but the service was extremely slow.",
        "Positive: Pasta & Ambiance": "The pasta was delicious and the ambiance was wonderful.",
        "Mixed: Management, Drinks vs Pizza": "Terrible management and overpriced drinks, though the pizza was okay.",
        "Negative: Chicken & Soup": "The chicken was raw and the soup was freezing cold.",
        "Mixed: Beer selection vs Atmosphere": "Great beer selection but the atmosphere was far too noisy.",
        "Negative: Table Wait & Hostess": "We waited 40 minutes for our table and the hostess was rude.",
        "Mixed: Dessert vs Prices": "Excellent dessert menu but the prices are exorbitant.",
        "Mixed: Sushi vs Bill": "The sushi was fresh, but the bill gave us a shock."
    }

    selected_example = st.selectbox("💡 Choose an example or write your own below:", list(examples.keys()))

    default_text = examples[selected_example] if selected_example != "Custom Review" else "The food was amazing but the service was extremely slow."

    user_input = st.text_area(
        "Enter Restaurant Review:",
        value=default_text,
        height=100,
        placeholder="Type a restaurant review (e.g. The food was tasty, but the service was terrible)..."
    )

    col_btn, col_clear = st.columns([1, 6])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Review", type="primary", use_container_width=True)

    if analyze_clicked or user_input:
        if not user_input.strip():
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing aspects and polarities..."):
                result = pipeline.analyze_review(user_input, classifier_type=model_code)

            st.markdown("### 📋 Analysis Results")

            # Overall Sentiment Badge
            overall = result["overall_sentiment"]
            overall_badge_class = (
                "sentiment-positive" if "Positive" in overall and "Mixed" not in overall else
                "sentiment-negative" if "Negative" in overall and "Mixed" not in overall else
                "sentiment-mixed" if "Mixed" in overall else
                "sentiment-neutral"
            )

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**Overall Review Polarity:** <span class='{overall_badge_class}'>{overall}</span>", unsafe_allow_html=True)
            with col2:
                st.metric("Detected Aspects", len(result["aspect_results"]))
            with col3:
                summary = result["sentiment_summary"]
                st.markdown(f"🟢 **{summary['positive']}** Pos | 🔴 **{summary['negative']}** Neg | ⚪ **{summary['neutral']}** Neu")

            st.markdown("---")

            # Per-Aspect Cards
            if not result["aspect_results"]:
                st.info("ℹ️ No specific restaurant aspect terms were detected in the text. The sentence was evaluated as a whole.")
            else:
                st.markdown("#### 🎯 Aspect-Level Breakdown")
                st.caption("Each aspect term is evaluated using its isolated local context window:")

                for item in result["aspect_results"]:
                    aspect_name = item["aspect"]
                    sentiment = item["sentiment"]
                    conf = item["confidence"]
                    probs = item["probabilities"]
                    ctx = item["context_used"]

                    badge_class = (
                        "sentiment-positive" if sentiment == "positive" else
                        "sentiment-negative" if sentiment == "negative" else
                        "sentiment-neutral"
                    )

                    with st.container():
                        st.markdown(f"""
                        <div class="aspect-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 1.15rem; font-weight: 700; color: #111827;">🏷️ {aspect_name.upper()}</span>
                                <span class="{badge_class}">{sentiment.upper()} ({conf*100:.1f}%)</span>
                            </div>
                            <div style="font-size: 0.95rem; color: #374151; margin-bottom: 8px;">
                                <strong>Isolated Local Context:</strong> <code>{ctx}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability Bar Breakdown
                        prob_cols = st.columns(3)
                        with prob_cols[0]:
                            st.progress(probs.get("positive", 0.0), text=f"Positive: {probs.get('positive', 0.0)*100:.1f}%")
                        with prob_cols[1]:
                            st.progress(probs.get("negative", 0.0), text=f"Negative: {probs.get('negative', 0.0)*100:.1f}%")
                        with prob_cols[2]:
                            st.progress(probs.get("neutral", 0.0), text=f"Neutral: {probs.get('neutral', 0.0)*100:.1f}%")
                        st.write("")


# ==========================================
# PAGE 2: DATASET & MODEL DASHBOARD
# ==========================================
elif page == "📊 Dataset & Model Dashboard":
    st.markdown('<div class="main-title">Dataset & Model Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Metrics, class distributions, and confusion matrices on SemEval-2014 Task 4 Gold Test Set.</div>', unsafe_allow_html=True)

    tab_eval, tab_data, tab_aspects = st.tabs(["📈 Model Evaluation & Confusion Matrix", "📊 Sentiment Class Distribution", "🏷️ Frequent Aspects Breakdown"])

    # TAB 1: MODEL EVALUATION
    with tab_eval:
        st.markdown("### 🏆 Gold Test Set Performance (N = 1,120 aspect instances)")

        lr_res = metrics_data.get("Logistic Regression", {})
        nb_res = metrics_data.get("Naive Bayes", {})

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("LR Test Accuracy", f"{lr_res.get('accuracy', 0)*100:.2f}%", f"+{(lr_res.get('accuracy', 0) - nb_res.get('accuracy', 0))*100:.2f}% vs NB")
        with col_m2:
            st.metric("LR Macro F1", f"{lr_res.get('macro_f1', 0)*100:.2f}%", f"+{(lr_res.get('macro_f1', 0) - nb_res.get('macro_f1', 0))*100:.2f}% vs NB")
        with col_m3:
            st.metric("LR Weighted F1", f"{lr_res.get('weighted_f1', 0)*100:.2f}%")
        with col_m4:
            st.metric("NB Baseline Accuracy", f"{nb_res.get('accuracy', 0)*100:.2f}%")

        st.markdown("#### 🔬 Detailed Metric Comparison")
        comp_df = pd.DataFrame({
            "Model": ["Logistic Regression (Balanced)", "Naive Bayes (Multinomial)"],
            "Accuracy": [f"{lr_res.get('accuracy', 0)*100:.2f}%", f"{nb_res.get('accuracy', 0)*100:.2f}%"],
            "Macro Precision": [f"{lr_res.get('macro_precision', 0)*100:.2f}%", f"{nb_res.get('macro_precision', 0)*100:.2f}%"],
            "Macro Recall": [f"{lr_res.get('macro_recall', 0)*100:.2f}%", f"{nb_res.get('macro_recall', 0)*100:.2f}%"],
            "Macro F1": [f"{lr_res.get('macro_f1', 0)*100:.2f}%", f"{nb_res.get('macro_f1', 0)*100:.2f}%"],
            "Weighted F1": [f"{lr_res.get('weighted_f1', 0)*100:.2f}%", f"{nb_res.get('weighted_f1', 0)*100:.2f}%"]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        st.markdown("#### 🔲 Confusion Matrices")
        labels = lr_res.get("labels", ["negative", "neutral", "positive"])

        fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # LR CM
        cm_lr = np.array(lr_res.get("confusion_matrix", [[0,0,0],[0,0,0],[0,0,0]]))
        sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title("Logistic Regression (Balanced)")
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        # NB CM
        cm_nb = np.array(nb_res.get("confusion_matrix", [[0,0,0],[0,0,0],[0,0,0]]))
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title("Naive Bayes (Baseline)")
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")

        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.caption("Observation for Viva: Naive Bayes suffers from extreme class-imbalance bias towards 'positive', whereas Logistic Regression with balanced class weights correctly identifies negative and neutral instances.")

    # TAB 2: SENTIMENT CLASS DISTRIBUTION
    with tab_data:
        st.markdown("### 📊 Dataset Sentiment Polarity Distribution")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("#### Training Set (N = 3,602)")
            train_counts = train_df["polarity"].value_counts()
            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
            colors = ["#10B981", "#EF4444", "#6B7280"]
            train_counts.plot(kind="bar", color=colors, ax=ax_tr)
            ax_tr.set_ylabel("Number of Aspect Terms")
            ax_tr.set_xlabel("Polarity")
            for p in ax_tr.patches:
                ax_tr.annotate(str(p.get_height()), (p.get_x() * 1.005 + 0.15, p.get_height() * 1.01))
            st.pyplot(fig_tr)
            plt.close(fig_tr)

        with col_d2:
            st.markdown("#### Testing Set (N = 1,120)")
            test_counts = test_df["polarity"].value_counts()
            fig_te, ax_te = plt.subplots(figsize=(6, 4))
            test_counts.plot(kind="bar", color=colors, ax=ax_te)
            ax_te.set_ylabel("Number of Aspect Terms")
            ax_te.set_xlabel("Polarity")
            for p in ax_te.patches:
                ax_te.annotate(str(p.get_height()), (p.get_x() * 1.005 + 0.15, p.get_height() * 1.01))
            st.pyplot(fig_te)
            plt.close(fig_te)

    # TAB 3: FREQUENT ASPECTS & ASPECT-WISE SENTIMENT
    with tab_aspects:
        st.markdown("### 🏷️ Top Frequent Restaurant Aspects in SemEval-2014")

        top_aspects = train_df["aspect_term"].str.lower().value_counts().head(12)

        fig_asp, ax_asp = plt.subplots(figsize=(10, 4.5))
        top_aspects.plot(kind="barh", color="#3B82F6", ax=ax_asp)
        ax_asp.invert_yaxis()
        ax_asp.set_xlabel("Frequency in Training Set")
        ax_asp.set_ylabel("Aspect Term")
        ax_asp.set_title("Top 12 Most Frequently Annotated Restaurant Aspects")
        for p in ax_asp.patches:
            ax_asp.annotate(f"{int(p.get_width())}", (p.get_width() + 4, p.get_y() + 0.5))
        st.pyplot(fig_asp)
        plt.close(fig_asp)

        st.markdown("#### ⚖️ Aspect-Wise Positive vs Negative Distribution")
        target_top = ["food", "service", "prices", "place", "staff", "menu", "pizza", "atmosphere"]
        aspect_sub = train_df[train_df["aspect_term"].str.lower().isin(target_top)]
        pivot_df = pd.crosstab(aspect_sub["aspect_term"].str.lower(), aspect_sub["polarity"])

        fig_piv, ax_piv = plt.subplots(figsize=(10, 4.5))
        pivot_df.plot(kind="bar", stacked=True, color={"positive": "#10B981", "negative": "#EF4444", "neutral": "#6B7280"}, ax=ax_piv)
        ax_piv.set_title("Polarity Distribution Across Key Restaurant Aspects")
        ax_piv.set_ylabel("Number of Mentions")
        ax_piv.set_xlabel("Aspect Term")
        plt.xticks(rotation=45)
        st.pyplot(fig_piv)
        plt.close(fig_piv)


# ==========================================
# PAGE 3: BATCH BENCHMARK TESTER
# ==========================================
elif page == "🧪 Batch Benchmark Tester":
    st.markdown('<div class="main-title">Batch Benchmark Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Run pre-defined test cases covering single, multiple, and mixed-sentiment reviews.</div>', unsafe_allow_html=True)

    from tests.test_reviews import TEST_REVIEWS

    if st.button("▶️ Run Batch Test on All 12 Reviews", type="primary"):
        results_list = []
        for item in TEST_REVIEWS:
            res = pipeline.analyze_review(item["review"], classifier_type=model_code)
            aspects_summary = ", ".join([f"{a['aspect']} ({a['sentiment'].upper()})" for a in res["aspect_results"]])
            results_list.append({
                "ID": item["id"],
                "Category": item["category"],
                "Review Text": item["review"],
                "Overall Sentiment": res["overall_sentiment"],
                "Extracted Aspects & Polarities": aspects_summary
            })

        st.success(f"Successfully evaluated {len(results_list)} benchmark reviews!")
        df_res = pd.DataFrame(results_list)
        st.dataframe(df_res, hide_index=True, use_container_width=True)
    else:
        st.markdown("Click the button above to execute automated evaluation across all 12 benchmark cases.")
        df_preview = pd.DataFrame([{"ID": r["id"], "Category": r["category"], "Review": r["review"]} for r in TEST_REVIEWS])
        st.dataframe(df_preview, hide_index=True, use_container_width=True)


# ==========================================
# PAGE 4: VIVA & ARCHITECTURE Q&A
# ==========================================
elif page == "📖 Viva & Architecture Q&A":
    st.markdown('<div class="main-title">College NLP Viva & Defense Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Theoretical explanations, architecture justifications, and honest system limitations.</div>', unsafe_allow_html=True)

    with st.expander("1. What is Aspect-Based Sentiment Analysis (ABSA) and how does it differ from Standard Sentiment Analysis?", expanded=True):
        st.write("""
        **Standard Sentiment Analysis** assigns a single global label to an entire review (e.g. 'Positive' or 'Negative').
        However, human reviews are naturally multidimensional:
        - *"The food was amazing but the service was extremely slow."*
        - A global sentiment classifier is forced to pick either positive or negative, losing critical granular feedback.
        - **ABSA decomposes reviews into pairs of (Aspect Term, Sentiment Polarity)**, recognizing that a customer can adore the cuisine while deploring the staff speed.
        """)

    with st.expander("2. Why did we choose Classical NLP (POS + TF-IDF + Logistic Regression) instead of an LLM API or BERT?"):
        st.write("""
        1. **Academic Understandability**: In a viva, every parameter and feature representation can be mathematically explained (Bayes' theorem, TF-IDF dot products, sigmoid probabilities).
        2. **Determinism & Reproducibility**: No external API rate limits, costs, network latency, or non-deterministic hallucinations.
        3. **Fast Local Execution**: The trained models load in milliseconds and run locally on standard CPU without GPU requirements.
        4. **Interpretability**: Feature weights reveal exact n-grams (e.g., *'extremely slow' -> negative weight*, *'top notch' -> positive weight*).
        """)

    with st.expander("3. How do we prevent feature bleeding in mixed-sentiment reviews?"):
        st.write("""
        This is solved through **Clause Boundary Splitting and Local Context Windowing**:
        - Adversative connectives (*but, however, yet, although, while*) and punctuation marks (*; , .*) serve as semantic split points.
        - When predicting sentiment for *'food'*, the classifier only examines the clause *'The food was amazing'*.
        - When predicting sentiment for *'service'*, it only examines *'the service was extremely slow'*.
        - The aspect term is concatenated with its clause context (`[aspect_term] + [local_context]`), generating isolated TF-IDF features.
        """)

    with st.expander("4. What are the known limitations of this system? (Honest Viva Discussion)"):
        st.write("""
        As required for an honest academic project:
        - **Implicit Aspects**: The system detects explicit aspects (nouns/noun phrases). Implicit aspects without explicit nouns (e.g., *"It cost an arm and a leg"* meaning price is high) require deeper semantic parsing.
        - **Complex Sarcasm**: Sarcastic remarks (*"Took only 3 hours to get water, great job!"*) require pragmatic modeling.
        - **Vocabulary Coverage**: Aspect extraction uses POS chunking combined with a 300-term domain lexicon. Unseen food names not tagged as nouns may require continuous domain vocabulary expansion.
        """)
