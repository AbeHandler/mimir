#!/usr/bin/env python3
"""Predict ATT using linguistic features extracted with spaCy."""

import json
import gzip
import numpy as np
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_att_results(filepath: Path) -> pd.DataFrame:
    """
    Load ATT results from gzipped JSONL file and expand to include both run1 and run3.

    Each sentence will generate up to 2 rows:
    - One row with att_run1 (if valid)
    - One row with att_run3 (if valid)

    Args:
        filepath: Path to sentence_att_results.jsonl.gz

    Returns:
        DataFrame with ATT analysis results, expanded for both runs
    """
    print(f"Loading ATT results from {filepath}")

    records = []
    with gzip.open(filepath, 'rt') as f:
        for line in tqdm(f, desc="Loading ATT data"):
            record = json.loads(line)

            # Create a row for run1 if ATT is valid
            if 'att_run1' in record and not pd.isna(record['att_run1']):
                run1_record = {
                    'id': record['id'] + '_run1',
                    'original_id': record['id'],
                    'text': record['text'],
                    'doc_id': record.get('doc_id', ''),
                    'att': record['att_run1'],
                    'loss_treated': record.get('loss_run1', np.nan),
                    'loss_control': record.get('loss_run4', np.nan),
                    'run': 'run1'
                }
                records.append(run1_record)

            # Create a row for run3 if ATT is valid
            if 'att_run3' in record and not pd.isna(record['att_run3']):
                run3_record = {
                    'id': record['id'] + '_run3',
                    'original_id': record['id'],
                    'text': record['text'],
                    'doc_id': record.get('doc_id', ''),
                    'att': record['att_run3'],
                    'loss_treated': record.get('loss_run3', np.nan),
                    'loss_control': record.get('loss_run4', np.nan),
                    'run': 'run3'
                }
                records.append(run3_record)

    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df):,} sentence-run pairs from both run1 and run3")
    print(f"  Run1 sentences: {(df['run'] == 'run1').sum():,}")
    print(f"  Run3 sentences: {(df['run'] == 'run3').sum():,}\n")

    return df


def extract_linguistic_features(text: str, nlp) -> Dict[str, Any]:
    """
    Extract linguistic features from text using spaCy.

    Args:
        text: Sentence text
        nlp: spaCy language model

    Returns:
        Dictionary of linguistic features
    """
    doc = nlp(text)

    features = {}

    # Length features
    features['char_length'] = len(text)
    features['token_count'] = len(doc)
    features['avg_token_length'] = np.mean([len(token.text) for token in doc]) if len(doc) > 0 else 0

    # POS tag distribution
    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = len(doc) if len(doc) > 0 else 1

    # Proportion of each POS tag
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 'NUM', 'PROPN']:
        features[f'pos_{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / total_tokens

    # Dependency features
    dep_counts = Counter([token.dep_ for token in doc])
    features['n_root'] = dep_counts.get('ROOT', 0)
    features['n_nsubj'] = dep_counts.get('nsubj', 0)
    features['n_dobj'] = dep_counts.get('dobj', 0)

    # Named entity features
    features['n_entities'] = len(doc.ents)
    features['entity_ratio'] = len(doc.ents) / total_tokens

    # Entity type distribution
    entity_types = Counter([ent.label_ for ent in doc.ents])
    for ent_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'CARDINAL']:
        features[f'entity_{ent_type.lower()}'] = entity_types.get(ent_type, 0)

    # Syntactic complexity
    features['max_depth'] = max([len(list(token.ancestors)) for token in doc]) if len(doc) > 0 else 0
    features['avg_depth'] = np.mean([len(list(token.ancestors)) for token in doc]) if len(doc) > 0 else 0

    # Lexical diversity
    unique_lemmas = len(set([token.lemma_.lower() for token in doc if token.is_alpha]))
    features['lexical_diversity'] = unique_lemmas / total_tokens if total_tokens > 0 else 0

    # Punctuation features
    features['n_punctuation'] = sum([1 for token in doc if token.is_punct])
    features['punct_ratio'] = features['n_punctuation'] / total_tokens

    # Stopword ratio
    features['stopword_ratio'] = sum([1 for token in doc if token.is_stop]) / total_tokens if total_tokens > 0 else 0

    # Capitalization features
    features['n_capitalized'] = sum([1 for token in doc if token.text and token.text[0].isupper()])
    features['cap_ratio'] = features['n_capitalized'] / total_tokens if total_tokens > 0 else 0

    # Sentence structure
    features['has_subordinate'] = any([token.dep_ in ['advcl', 'acl', 'relcl'] for token in doc])
    features['has_coordination'] = any([token.dep_ == 'cc' for token in doc])

    # Number features
    features['has_numbers'] = any([token.like_num for token in doc])
    features['n_numbers'] = sum([1 for token in doc if token.like_num])

    # URL and special tokens
    features['has_url'] = any([token.like_url for token in doc])
    features['has_email'] = any([token.like_email for token in doc])

    return features


def extract_bow_features(texts: List[str], nlp, max_features: int = 100) -> pd.DataFrame:
    """
    Extract Bag of Words features using lemmatized tokens.

    Args:
        texts: List of sentence texts
        nlp: spaCy language model
        max_features: Maximum number of top tokens to include

    Returns:
        DataFrame with BOW features
    """
    print("\nExtracting Bag of Words features...")

    # Count all lemmas
    all_lemmas = []
    for text in tqdm(texts, desc="Lemmatizing"):
        doc = nlp(text)
        lemmas = [token.lemma_.lower() for token in doc
                  if token.is_alpha and not token.is_stop and len(token.text) > 2]
        all_lemmas.extend(lemmas)

    # Get top N most common lemmas
    lemma_counts = Counter(all_lemmas)
    top_lemmas = [lemma for lemma, _ in lemma_counts.most_common(max_features)]

    print(f"Using top {len(top_lemmas)} lemmas as BOW features")

    # Create BOW features
    bow_features = []
    for text in tqdm(texts, desc="Creating BOW features"):
        doc = nlp(text)
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        lemma_set = set(lemmas)

        bow = {f'bow_{lemma}': 1 if lemma in lemma_set else 0 for lemma in top_lemmas}
        bow_features.append(bow)

    return pd.DataFrame(bow_features)


def create_feature_matrix(df: pd.DataFrame, sample_size: int = 20000) -> pd.DataFrame:
    """
    Create feature matrix from sentences.

    Args:
        df: DataFrame with ATT results (includes both run1 and run3)
        sample_size: Number of sentence-run pairs to sample for analysis

    Returns:
        DataFrame with all features
    """
    # Sample if dataset is large
    if len(df) > sample_size:
        print(f"Sampling {sample_size:,} sentence-run pairs from {len(df):,} total")
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()

    print(f"\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=[])
    print("✓ spaCy model loaded\n")

    # Extract linguistic features
    print("Extracting linguistic features...")
    linguistic_features = []
    for text in tqdm(df_sample['text'], desc="Processing sentences", unit="sent"):
        features = extract_linguistic_features(text, nlp)
        linguistic_features.append(features)

    linguistic_df = pd.DataFrame(linguistic_features)

    # Extract BOW features
    bow_df = extract_bow_features(df_sample['text'].tolist(), nlp, max_features=50)

    # Combine all features
    feature_df = pd.concat([
        df_sample[['id', 'original_id', 'att', 'loss_treated', 'loss_control', 'run']].reset_index(drop=True),
        linguistic_df,
        bow_df
    ], axis=1)

    print(f"\n✓ Created feature matrix with {len(feature_df.columns)} features")

    return feature_df


def train_att_predictor(feature_df: pd.DataFrame):
    """
    Train linear regression to predict ATT (continuous variable).

    Args:
        feature_df: DataFrame with features and ATT values
    """
    print(f"\n{'='*80}")
    print(f"TRAINING ATT PREDICTOR (Linear Regression)")
    print(f"{'='*80}\n")

    # ATT statistics
    print(f"ATT Statistics:")
    print(f"  Mean: {feature_df['att'].mean():.6f}")
    print(f"  Median: {feature_df['att'].median():.6f}")
    print(f"  Std: {feature_df['att'].std():.6f}")
    print(f"  Min: {feature_df['att'].min():.6f}")
    print(f"  Max: {feature_df['att'].max():.6f}\n")

    # Select features (exclude id, att, losses, and run info)
    exclude_cols = ['id', 'original_id', 'att', 'loss_treated', 'loss_control', 'run']
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]

    X = feature_df[feature_cols]
    y = feature_df['att']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples\n")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train linear regression
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n{'='*80}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*80}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"\nTraining RMSE: {train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"\nTraining MAE: {train_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}\n")

    # Feature importance
    print(f"\n{'='*80}")
    print(f"TOP FEATURES PREDICTING ATT")
    print(f"{'='*80}\n")

    # Get feature coefficients
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_
    })

    # Sort by absolute coefficient
    feature_importance['abs_coef'] = feature_importance['coefficient'].abs()
    feature_importance = feature_importance.sort_values('abs_coef', ascending=False)

    print("Top 20 features by absolute coefficient magnitude:")
    print("-" * 70)
    print(f"{'Feature':40s} {'Coefficient':>12s} {'Direction':>15s}")
    print("-" * 70)

    for idx, row in feature_importance.head(20).iterrows():
        direction = "↑ Higher ATT" if row['coefficient'] > 0 else "↓ Lower ATT"
        print(f"{row['feature']:40s} {row['coefficient']:12.6f} {direction:>15s}")

    print("\n\nFeatures predicting POSITIVE ATT (top 10):")
    print("-" * 60)
    print("These features increase ATT (treated has higher loss than control)")
    print("-" * 60)
    positive_features = feature_importance[feature_importance['coefficient'] > 0].head(10)
    for idx, row in positive_features.iterrows():
        print(f"{row['feature']:40s} {row['coefficient']:10.6f}")

    print("\n\nFeatures predicting NEGATIVE ATT (top 10):")
    print("-" * 60)
    print("These features decrease ATT (treated has lower loss than control)")
    print("-" * 60)
    negative_features = feature_importance[feature_importance['coefficient'] < 0].head(10)
    for idx, row in negative_features.iterrows():
        print(f"{row['feature']:40s} {row['coefficient']:10.6f}")

    # Save feature importance
    output_file = Path("/tmp/att_feature_importance.csv")
    feature_importance.to_csv(output_file, index=False)
    print(f"\n✓ Saved feature importance to {output_file}")

    # Scatter plot data for manual inspection
    results_df = pd.DataFrame({
        'true_att': y_test,
        'predicted_att': y_test_pred,
        'residual': y_test - y_test_pred
    })
    results_file = Path("/tmp/att_predictions.csv")
    results_df.to_csv(results_file, index=False)
    print(f"✓ Saved predictions to {results_file}")

    return model, scaler, feature_cols, feature_importance


def main():
    """Main entry point."""
    att_results_file = Path("/tmp/sentence_att_results.jsonl.gz")

    if not att_results_file.exists():
        print(f"Error: {att_results_file} not found")
        print("Please run analyze_sentences.py first to generate ATT results")
        return

    # Load ATT results
    df = load_att_results(att_results_file)

    # Remove rows with NaN ATT
    df = df[~df['att'].isna()].copy()
    print(f"Valid sentence-run pairs with ATT: {len(df):,}\n")

    # Create feature matrix (using 20k samples to capture data from both runs)
    feature_df = create_feature_matrix(df, sample_size=20000)

    # Train predictor
    model, scaler, feature_cols, feature_importance = train_att_predictor(feature_df)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
