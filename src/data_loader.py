"""
data_loader.py
==============
Module responsible for loading and parsing the SemEval-2014 Task 4
Restaurant Reviews XML datasets into structured Pandas DataFrames.

Academic Explanation:
---------------------
SemEval-2014 Task 4 represents the gold-standard benchmark for Aspect-Based
Sentiment Analysis (ABSA). Unlike traditional sentence-level sentiment analysis
where a review is treated as a single label, ABSA requires identifying:
1. Aspect Terms: Specific entities or targets (e.g., 'food', 'service', 'crust').
2. Aspect Polarities: The sentiment directed specifically toward that aspect
   ('positive', 'negative', 'neutral', 'conflict').

Here, we parse the nested XML tree, handle character offsets, and produce
clean tabular records suitable for classical machine learning pipelines.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import pandas as pd


def parse_semeval_xml(xml_path: str, include_conflict: bool = False) -> pd.DataFrame:
    """
    Parses a SemEval-2014 XML file and returns a DataFrame of aspect annotations.

    Parameters:
    -----------
    xml_path : str
        Path to the SemEval XML file (train or test).
    include_conflict : bool
        Whether to retain 'conflict' polarity examples (default False, as
        standard 3-way ABSA benchmarks evaluate on pos, neg, neu).

    Returns:
    --------
    pd.DataFrame:
        DataFrame with columns:
        ['sentence_id', 'text', 'aspect_term', 'polarity', 'from_idx', 'to_idx']
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"SemEval XML file not found at: {xml_path}")

    # Step 1: Parse the XML document tree
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []

    # Step 2: Iterate over every <sentence> node
    for sentence in root.findall("sentence"):
        s_id = sentence.attrib.get("id", "")
        text_elem = sentence.find("text")
        if text_elem is None or not text_elem.text:
            continue
        text = text_elem.text.strip()

        # Step 3: Extract nested <aspectTerms>
        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            for aspect in aspect_terms.findall("aspectTerm"):
                term = aspect.attrib.get("term", "").strip()
                polarity = aspect.attrib.get("polarity", "").lower().strip()
                from_idx = int(aspect.attrib.get("from", 0))
                to_idx = int(aspect.attrib.get("to", 0))

                # Academic standard: Focus on 3 main sentiment polarities
                if not include_conflict and polarity == "conflict":
                    continue

                if polarity in ["positive", "negative", "neutral"]:
                    records.append({
                        "sentence_id": s_id,
                        "text": text,
                        "aspect_term": term,
                        "polarity": polarity,
                        "from_idx": from_idx,
                        "to_idx": to_idx
                    })

    df = pd.DataFrame(records)
    return df


def load_dataset(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads both training and testing datasets from the raw XML directory.
    If cached CSV files exist in processed/, loads from CSV for fast access.
    Otherwise, parses XML and saves the processed CSVs.

    Parameters:
    -----------
    data_dir : str
        Base data directory containing 'raw/' and 'processed/' subfolders.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        (train_df, test_df)
    """
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)

    train_csv = os.path.join(proc_dir, "train_aspects.csv")
    test_csv = os.path.join(proc_dir, "test_aspects.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print(f"Loading cached datasets from {proc_dir}...")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        return train_df, test_df

    train_xml = os.path.join(raw_dir, "Restaurants_Train_v2.xml")
    test_xml = os.path.join(raw_dir, "Restaurants_Test_Gold.xml")

    print(f"Parsing raw SemEval training XML: {train_xml}...")
    train_df = parse_semeval_xml(train_xml, include_conflict=False)
    train_df.to_csv(train_csv, index=False)
    print(f"Saved {len(train_df)} training aspect instances to {train_csv}")

    print(f"Parsing raw SemEval testing XML: {test_xml}...")
    test_df = parse_semeval_xml(test_xml, include_conflict=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved {len(test_df)} testing aspect instances to {test_csv}")

    return train_df, test_df


def extract_aspect_lexicon(train_df: pd.DataFrame, min_freq: int = 2) -> Dict[str, int]:
    """
    Builds a high-precision domain lexicon of restaurant aspect terms
    from the training annotations. Used during inference for candidate matching.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame containing 'aspect_term'.
    min_freq : int
        Minimum occurrence count to include an aspect term.

    Returns:
    --------
    Dict[str, int]:
        Dictionary mapping lowercased aspect terms to their frequency.
    """
    counts = train_df["aspect_term"].str.lower().value_counts()
    frequent_aspects = counts[counts >= min_freq].to_dict()
    return frequent_aspects


if __name__ == "__main__":
    # Self-test script when executed directly
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    train, test = load_dataset(base_dir)
    print("\nDataset Summary:")
    print(f"Training aspects: {len(train)}")
    print(train["polarity"].value_counts())
    print(f"\nTesting aspects: {len(test)}")
    print(test["polarity"].value_counts())
