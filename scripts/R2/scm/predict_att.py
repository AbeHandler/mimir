#!/usr/bin/env python3
"""Predict ATT using SBERT embeddings for individual sentences."""

import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def compute_token_ll_stats(tokens: List[Dict]) -> Dict[str, float]:
    """
    Compute statistics from token-level log-likelihoods.

    Args:
        tokens: List of token dictionaries with 'log_prob' field

    Returns:
        Dictionary with LL statistics: min, max, mean, median, std, q25, q75
    """
    if not tokens:
        return {
            'll_min': np.nan, 'll_max': np.nan, 'll_mean': np.nan,
            'll_median': np.nan, 'll_std': np.nan, 'll_q25': np.nan, 'll_q75': np.nan,
            'll_count': 0
        }

    log_probs = []
    for token in tokens:
        log_prob = token.get('log_prob')
        if log_prob is not None and not np.isnan(log_prob):
            log_probs.append(log_prob)

    if not log_probs:
        return {
            'll_min': np.nan, 'll_max': np.nan, 'll_mean': np.nan,
            'll_median': np.nan, 'll_std': np.nan, 'll_q25': np.nan, 'll_q75': np.nan,
            'll_count': 0
        }

    log_probs_arr = np.array(log_probs)

    return {
        'll_min': np.min(log_probs_arr),
        'll_max': np.max(log_probs_arr),
        'll_mean': np.mean(log_probs_arr),
        'll_median': np.median(log_probs_arr),
        'll_std': np.std(log_probs_arr),
        'll_q25': np.percentile(log_probs_arr, 25),
        'll_q75': np.percentile(log_probs_arr, 75),
        'll_count': len(log_probs)
    }


def load_token_data(filepath: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load token-level data from merged sentences file.

    Args:
        filepath: Path to merged_sentences.jsonl.gz

    Returns:
        Dictionary mapping sentence ID to run data with tokens:
        {
            'sentence_id': {
                'pair1_treated_run1': {'tokens': [...]},
                'pair2_treated_run3': {'tokens': [...]},
                'pair2_control_run4': {'tokens': [...]}
            }
        }
    """
    print(f"Loading token-level data from {filepath}")

    token_data = {}

    with gzip.open(filepath, 'rt') as f:
        for line in tqdm(f, desc="Loading token data"):
            record = json.loads(line)
            sent_id = record.get('id')

            if sent_id:
                token_data[sent_id] = {
                    'pair1_treated_run1': record.get('pair1_treated_run1'),
                    'pair2_treated_run3': record.get('pair2_treated_run3'),
                    'pair2_control_run4': record.get('pair2_control_run4')
                }

    print(f"✓ Loaded token data for {len(token_data):,} sentences\n")
    return token_data


def load_att_results(filepath: Path, token_data: Dict[str, Dict[str, List[Dict]]] = None) -> pd.DataFrame:
    """
    Load ATT results from gzipped JSONL file and expand to include both run1 and run3.

    Each sentence will generate up to 2 rows:
    - One row with att_run1 (if valid)
    - One row with att_run3 (if valid)

    Args:
        filepath: Path to sentence_att_results.jsonl.gz
        token_data: Optional dictionary with token-level data for computing LL features

    Returns:
        DataFrame with ATT analysis results, expanded for both runs
    """
    print(f"Loading ATT results from {filepath}")

    records = []
    embeddings_found = 0
    embeddings_missing = 0
    ll_features_computed = 0

    with gzip.open(filepath, 'rt') as f:
        for line in tqdm(f, desc="Loading ATT data"):
            record = json.loads(line)
            sent_id = record['id']

            # Extract embedding if present
            embedding = record.get('embedding')
            if embedding is not None:
                embeddings_found += 1
            else:
                embeddings_missing += 1

            # Get token-level data for LL features if available
            # Only use R1 and R3 features to avoid perfectly predicting ATT
            ll_treated_run1 = {}
            ll_treated_run3 = {}

            if token_data and sent_id in token_data:
                sent_token_data = token_data[sent_id]

                if sent_token_data.get('pair1_treated_run1'):
                    tokens = sent_token_data['pair1_treated_run1'].get('tokens', [])
                    ll_treated_run1 = compute_token_ll_stats(tokens)
                    ll_features_computed += 1

                if sent_token_data.get('pair2_treated_run3'):
                    tokens = sent_token_data['pair2_treated_run3'].get('tokens', [])
                    ll_treated_run3 = compute_token_ll_stats(tokens)

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
                    'run': 'run1',
                    'sentence_number': record.get('sentence_number', -1),
                    'sentence_position': record.get('sentence_position', np.nan),
                    'embedding': embedding
                }

                # Add LL features for treated (run1) only - no R4 features
                for key, val in ll_treated_run1.items():
                    run1_record[f'treated_{key}'] = val

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
                    'run': 'run3',
                    'sentence_number': record.get('sentence_number', -1),
                    'sentence_position': record.get('sentence_position', np.nan),
                    'embedding': embedding
                }

                # Add LL features for treated (run3) only - no R4 features
                for key, val in ll_treated_run3.items():
                    run3_record[f'treated_{key}'] = val

                records.append(run3_record)

    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df):,} sentence-run pairs from both run1 and run3")
    print(f"  Run1 sentences: {(df['run'] == 'run1').sum():,}")
    print(f"  Run3 sentences: {(df['run'] == 'run3').sum():,}")
    print(f"  Sentences with embeddings: {embeddings_found:,}")
    print(f"  Sentences without embeddings: {embeddings_missing:,}")
    if token_data:
        print(f"  Sentences with LL features: {ll_features_computed:,}")
    print()

    return df




def create_feature_matrix(df: pd.DataFrame, sample_size: int = 120000) -> pd.DataFrame:
    """
    Create feature matrix from sentences using SBERT embeddings and LL features.

    Args:
        df: DataFrame with ATT results (includes both run1 and run3)
        sample_size: Number of sentence-run pairs to sample for analysis

    Returns:
        DataFrame with SBERT embedding features, LL features, and metadata
    """
    # Sample if dataset is large
    if len(df) > sample_size:
        print(f"Sampling {sample_size:,} sentence-run pairs from {len(df):,} total")
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()

    # Identify LL feature columns
    ll_feature_cols = [col for col in df_sample.columns if col.startswith('treated_ll_') or col.startswith('control_ll_')]
    if ll_feature_cols:
        print(f"\nFound {len(ll_feature_cols)} LL feature columns")
        print(f"  Example LL features: {ll_feature_cols[:5]}")
        ll_features_available = df_sample[ll_feature_cols[0]].notna().sum()
        print(f"  Rows with LL features: {ll_features_available:,} / {len(df_sample):,}")

    # Process SBERT embeddings into separate feature columns
    print("\nProcessing SBERT embeddings...")
    embeddings_with_data = df_sample['embedding'].notna().sum()
    print(f"  Rows with embeddings: {embeddings_with_data:,} / {len(df_sample):,}")

    if embeddings_with_data > 0:
        # Get first non-null embedding to determine dimensionality
        first_embedding = df_sample[df_sample['embedding'].notna()]['embedding'].iloc[0]
        embedding_dim = len(first_embedding)
        print(f"  SBERT embedding dimension: {embedding_dim}")

        # Create embedding features
        embedding_data = []
        for emb in df_sample['embedding']:
            if emb is not None and isinstance(emb, list):
                embedding_data.append(emb)
            else:
                # Fill missing embeddings with zeros
                embedding_data.append([0.0] * embedding_dim)

        embedding_df = pd.DataFrame(
            embedding_data,
            columns=[f'sbert_{i}' for i in range(embedding_dim)]
        )
        print(f"✓ Created {embedding_dim} SBERT embedding features")
    else:
        print("  ERROR: No SBERT embeddings found!")
        raise ValueError("No SBERT embeddings available for prediction. Please run compute_sbert_embeddings.py first.")

    # Combine metadata, LL features, and embeddings
    metadata_cols = ['id', 'original_id', 'att', 'loss_treated', 'loss_control', 'run', 'sentence_number', 'sentence_position']
    metadata_cols.extend(ll_feature_cols)

    feature_df = pd.concat([
        df_sample[metadata_cols].reset_index(drop=True),
        embedding_df
    ], axis=1)

    total_features = embedding_dim + len(ll_feature_cols) + 2  # +2 for sentence_number and sentence_position
    print(f"\n✓ Created feature matrix with {len(feature_df.columns)} total columns")
    print(f"  - {embedding_dim} SBERT features")
    print(f"  - {len(ll_feature_cols)} LL features")
    print(f"  - 2 position features (sentence_number, sentence_position)")

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
    # Note: sentence_number and sentence_position are now included as features
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
    merged_sentences_file = Path("/tmp/merged_sentences.jsonl.gz")

    if not att_results_file.exists():
        print(f"Error: {att_results_file} not found")
        print("Please run analyze_sentences.py first to generate ATT results")
        return

    # Load token-level data for LL features
    token_data = None
    if merged_sentences_file.exists():
        print(f"\n{'='*80}")
        print(f"LOADING TOKEN-LEVEL DATA FOR LL FEATURES")
        print(f"{'='*80}\n")
        token_data = load_token_data(merged_sentences_file)
    else:
        print(f"\nWarning: {merged_sentences_file} not found")
        print("Proceeding without LL features. Run analyze_sentences.py to generate merged data.\n")

    # Load ATT results with LL features
    df = load_att_results(att_results_file, token_data=token_data)

    # Remove rows with NaN ATT
    df = df[~df['att'].isna()].copy()
    print(f"Valid sentence-run pairs with ATT: {len(df):,}\n")

    # Create feature matrix (using 120k samples to capture data from both runs)
    feature_df = create_feature_matrix(df, sample_size=120000)

    # Train predictor
    model, scaler, feature_cols, feature_importance = train_att_predictor(feature_df)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
