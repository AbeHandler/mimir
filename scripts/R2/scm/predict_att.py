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
    embeddings_found = 0
    embeddings_missing = 0

    with gzip.open(filepath, 'rt') as f:
        for line in tqdm(f, desc="Loading ATT data"):
            record = json.loads(line)

            # Extract embedding if present
            embedding = record.get('embedding')
            if embedding is not None:
                embeddings_found += 1
            else:
                embeddings_missing += 1

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
                records.append(run3_record)

    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df):,} sentence-run pairs from both run1 and run3")
    print(f"  Run1 sentences: {(df['run'] == 'run1').sum():,}")
    print(f"  Run3 sentences: {(df['run'] == 'run3').sum():,}")
    print(f"  Sentences with embeddings: {embeddings_found:,}")
    print(f"  Sentences without embeddings: {embeddings_missing:,}\n")

    return df




def create_feature_matrix(df: pd.DataFrame, sample_size: int = 50000) -> pd.DataFrame:
    """
    Create feature matrix from sentences using SBERT embeddings only.

    Args:
        df: DataFrame with ATT results (includes both run1 and run3)
        sample_size: Number of sentence-run pairs to sample for analysis

    Returns:
        DataFrame with SBERT embedding features and metadata
    """
    # Sample if dataset is large
    if len(df) > sample_size:
        print(f"Sampling {sample_size:,} sentence-run pairs from {len(df):,} total")
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()

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

    # Combine metadata and embeddings
    feature_df = pd.concat([
        df_sample[['id', 'original_id', 'att', 'loss_treated', 'loss_control', 'run', 'sentence_number', 'sentence_position']].reset_index(drop=True),
        embedding_df
    ], axis=1)

    print(f"\n✓ Created feature matrix with {len(feature_df.columns)} total columns ({embedding_dim} SBERT features)")

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

    if not att_results_file.exists():
        print(f"Error: {att_results_file} not found")
        print("Please run analyze_sentences.py first to generate ATT results")
        return

    # Load ATT results
    df = load_att_results(att_results_file)

    # Remove rows with NaN ATT
    df = df[~df['att'].isna()].copy()
    print(f"Valid sentence-run pairs with ATT: {len(df):,}\n")

    # Create feature matrix (using 50k samples to capture data from both runs)
    feature_df = create_feature_matrix(df, sample_size=50000)

    # Train predictor
    model, scaler, feature_cols, feature_importance = train_att_predictor(feature_df)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
