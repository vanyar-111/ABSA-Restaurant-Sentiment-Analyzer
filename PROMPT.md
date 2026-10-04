\# Project Prompt



Design and implement a complete Aspect-Based Sentiment Analysis system for

restaurant reviews using the SemEval-2014 Task 4 Restaurant Reviews dataset.



The system should:



\- preprocess restaurant review text using NLP techniques;

\- extract explicit restaurant aspects using POS tagging, noun/compound-noun

&#x20; patterns, and a restaurant-domain aspect vocabulary;

\- isolate local context around each aspect to reduce sentiment leakage between

&#x20; different aspects in the same review;

\- represent aspect-context pairs using TF-IDF features;

\- train Logistic Regression as the primary sentiment classifier;

\- train Multinomial Naive Bayes as a baseline classifier;

\- classify each aspect as positive, negative, or neutral;

\- aggregate aspect-level predictions into an overall review polarity;

\- evaluate the classifiers using accuracy, macro precision, macro recall,

&#x20; macro F1, weighted F1, per-class metrics, and confusion matrices;

\- provide a Streamlit interface for interactive review analysis and model

&#x20; evaluation;

\- include the trained model artifacts and all files required to run the system.



The implementation should use Python and standard NLP/machine-learning

libraries and should be reproducible from the files included in the

repository.

