#!/usr/bin/env python3
"""Predict ATT at the token level using token features."""

import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import spacy
    from spacy.tokens import Doc, Token
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("Warning: spaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    nlp = None


def compute_sentence_ll_stats(tokens: List[Dict]) -> Dict[str, float]:
    """
    Compute sentence-level statistics from token log-likelihoods.

    Args:
        tokens: List of token dictionaries with 'log_prob' field

    Returns:
        Dictionary with sentence-level LL statistics
    """
    if not tokens:
        return {
            'sent_ll_min': np.nan, 'sent_ll_max': np.nan, 'sent_ll_mean': np.nan,
            'sent_ll_median': np.nan, 'sent_ll_std': np.nan, 'sent_ll_q25': np.nan,
            'sent_ll_q75': np.nan, 'sent_ll_count': 0
        }

    log_probs = []
    for token in tokens:
        log_prob = token.get('log_prob')
        if log_prob is not None and not np.isnan(log_prob):
            log_probs.append(log_prob)

    if not log_probs:
        return {
            'sent_ll_min': np.nan, 'sent_ll_max': np.nan, 'sent_ll_mean': np.nan,
            'sent_ll_median': np.nan, 'sent_ll_std': np.nan, 'sent_ll_q25': np.nan,
            'sent_ll_q75': np.nan, 'sent_ll_count': 0
        }

    log_probs_arr = np.array(log_probs)

    return {
        'sent_ll_min': np.min(log_probs_arr),
        'sent_ll_max': np.max(log_probs_arr),
        'sent_ll_mean': np.mean(log_probs_arr),
        'sent_ll_median': np.median(log_probs_arr),
        'sent_ll_std': np.std(log_probs_arr),
        'sent_ll_q25': np.percentile(log_probs_arr, 25),
        'sent_ll_q75': np.percentile(log_probs_arr, 75),
        'sent_ll_count': len(log_probs)
    }


def extract_pos_features(spacy_token: Optional[Token]) -> Dict[str, Any]:
    """
    Extract part-of-speech features from a spaCy token.

    Args:
        spacy_token: spaCy Token object (or None if spaCy not available)

    Returns:
        Dictionary with POS features
    """
    if spacy_token is None:
        return {
            'pos': 'UNK',
            'tag': 'UNK',
            'is_punct': False,
            'is_stop': False,
            'is_alpha': False,
            'is_digit': False,
            'is_space': False,
            'is_ascii': False,
            'is_lower': False,
            'is_upper': False,
            'is_title': False,
            'like_num': False,
            'like_url': False,
            'like_email': False,
        }

    return {
        'pos': spacy_token.pos_,
        'tag': spacy_token.tag_,
        'is_punct': spacy_token.is_punct,
        'is_stop': spacy_token.is_stop,
        'is_alpha': spacy_token.is_alpha,
        'is_digit': spacy_token.is_digit,
        'is_space': spacy_token.is_space,
        'is_ascii': spacy_token.is_ascii,
        'is_lower': spacy_token.is_lower,
        'is_upper': spacy_token.is_upper,
        'is_title': spacy_token.is_title,
        'like_num': spacy_token.like_num,
        'like_url': spacy_token.like_url,
        'like_email': spacy_token.like_email,
    }


def extract_dependency_features(spacy_token: Optional[Token]) -> Dict[str, Any]:
    """
    Extract dependency parse features from a spaCy token.

    Args:
        spacy_token: spaCy Token object (or None if spaCy not available)

    Returns:
        Dictionary with dependency features
    """
    if spacy_token is None:
        return {
            'dep': 'UNK',
            'head_pos': 'UNK',
            'head_dep': 'UNK',
            'n_lefts': 0,
            'n_rights': 0,
            'is_sent_start': False,
            'is_sent_end': False,
            'head_distance': 0,
            'is_root': False,
            'is_child_of_num': False,
            'is_child_of_verb': False,
            'is_child_of_noun': False,
        }

    # Basic dependency info
    dep = spacy_token.dep_
    head = spacy_token.head
    head_pos = head.pos_ if head != spacy_token else 'ROOT'
    head_dep = head.dep_ if head != spacy_token else 'ROOT'

    # Structural features
    n_lefts = len(list(spacy_token.lefts))
    n_rights = len(list(spacy_token.rights))

    # Position in sentence
    is_sent_start = spacy_token.is_sent_start if spacy_token.is_sent_start is not None else False
    is_sent_end = spacy_token.i == len(spacy_token.doc) - 1

    # Distance to head (for dependency tree depth)
    head_distance = abs(spacy_token.i - head.i) if head != spacy_token else 0

    # Is this token the root of the sentence?
    is_root = spacy_token == head

    # Is this token a child of specific POS tags?
    is_child_of_num = head.pos_ == 'NUM' if head != spacy_token else False
    is_child_of_verb = head.pos_ == 'VERB' if head != spacy_token else False
    is_child_of_noun = head.pos_ in ('NOUN', 'PROPN') if head != spacy_token else False

    return {
        'dep': dep,
        'head_pos': head_pos,
        'head_dep': head_dep,
        'n_lefts': n_lefts,
        'n_rights': n_rights,
        'is_sent_start': is_sent_start,
        'is_sent_end': is_sent_end,
        'head_distance': head_distance,
        'is_root': is_root,
        'is_child_of_num': is_child_of_num,
        'is_child_of_verb': is_child_of_verb,
        'is_child_of_noun': is_child_of_noun,
    }


def extract_entity_features(spacy_token: Optional[Token]) -> Dict[str, Any]:
    """
    Extract named entity features from a spaCy token.

    Args:
        spacy_token: spaCy Token object (or None if spaCy not available)

    Returns:
        Dictionary with entity features
    """
    if spacy_token is None:
        return {
            'is_entity': False,
            'entity_type': 'NONE',
            'entity_iob': 'O',
            'is_person': False,
            'is_org': False,
            'is_gpe': False,
            'is_date': False,
            'is_time': False,
            'is_money': False,
            'is_quantity': False,
            'is_cardinal': False,
            'is_ordinal': False,
        }

    # Entity information
    ent_type = spacy_token.ent_type_ if spacy_token.ent_type_ else 'NONE'
    ent_iob = spacy_token.ent_iob_
    is_entity = ent_iob in ('B', 'I')

    # Specific entity types
    is_person = ent_type == 'PERSON'
    is_org = ent_type == 'ORG'
    is_gpe = ent_type == 'GPE'  # Geopolitical entity
    is_date = ent_type == 'DATE'
    is_time = ent_type == 'TIME'
    is_money = ent_type == 'MONEY'
    is_quantity = ent_type == 'QUANTITY'
    is_cardinal = ent_type == 'CARDINAL'  # Numbers
    is_ordinal = ent_type == 'ORDINAL'  # First, second, etc.

    return {
        'is_entity': is_entity,
        'entity_type': ent_type,
        'entity_iob': ent_iob,
        'is_person': is_person,
        'is_org': is_org,
        'is_gpe': is_gpe,
        'is_date': is_date,
        'is_time': is_time,
        'is_money': is_money,
        'is_quantity': is_quantity,
        'is_cardinal': is_cardinal,
        'is_ordinal': is_ordinal,
    }


def extract_token_features(
    token_idx: int,
    tokens: List[Dict],
    sentence_stats: Dict[str, float],
    spacy_doc: Optional[Doc] = None
) -> Dict[str, Any]:
    """
    Extract features for a single token.

    Args:
        token_idx: Index of token in sentence
        tokens: All tokens in the sentence
        sentence_stats: Sentence-level LL statistics
        spacy_doc: Optional spaCy Doc object for linguistic features

    Returns:
        Dictionary with token-level features
    """
    token = tokens[token_idx]
    num_tokens = len(tokens)

    # Token's own log-likelihood
    token_ll = token.get('log_prob', np.nan)

    # Position features
    position_abs = token_idx  # 0-indexed position
    position_rel = token_idx / max(num_tokens - 1, 1)  # Relative position [0, 1]

    # Context window features (3 tokens before and after)
    window_lls = []
    for offset in range(-3, 4):
        if offset == 0:
            continue
        ctx_idx = token_idx + offset
        if 0 <= ctx_idx < num_tokens:
            ctx_ll = tokens[ctx_idx].get('log_prob')
            if ctx_ll is not None and not np.isnan(ctx_ll):
                window_lls.append(ctx_ll)

    # Context statistics
    if window_lls:
        context_ll_mean = np.mean(window_lls)
        context_ll_std = np.std(window_lls)
        context_ll_min = np.min(window_lls)
        context_ll_max = np.max(window_lls)
    else:
        context_ll_mean = np.nan
        context_ll_std = np.nan
        context_ll_min = np.nan
        context_ll_max = np.nan

    # Relative to sentence statistics
    sent_ll_mean = sentence_stats.get('sent_ll_mean', np.nan)
    if not np.isnan(token_ll) and not np.isnan(sent_ll_mean):
        ll_deviation_from_sent_mean = token_ll - sent_ll_mean
    else:
        ll_deviation_from_sent_mean = np.nan

    # Extract linguistic features from spaCy
    spacy_token = None
    if spacy_doc is not None and token_idx < len(spacy_doc):
        spacy_token = spacy_doc[token_idx]

    pos_features = extract_pos_features(spacy_token)
    dep_features = extract_dependency_features(spacy_token)
    ent_features = extract_entity_features(spacy_token)

    return {
        'token_text': token.get('text', ''),
        'token_ll': token_ll,
        'position_abs': position_abs,
        'position_rel': position_rel,
        'num_tokens': num_tokens,
        'context_ll_mean': context_ll_mean,
        'context_ll_std': context_ll_std,
        'context_ll_min': context_ll_min,
        'context_ll_max': context_ll_max,
        'll_deviation_from_sent_mean': ll_deviation_from_sent_mean,
        **{f'sent_{k}': v for k, v in sentence_stats.items()},  # Add sentence-level stats
        **pos_features,  # Add POS features
        **dep_features,  # Add dependency features
        **ent_features,  # Add entity features
    }


def load_token_level_data(filepath: Path, sample_size: int = 100000) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and process token-level data with ATT calculation.

    Each token gets its own row with:
    - Token-level ATT (treated_ll - control_ll for this specific token)
    - Token features (position, own LL, context)
    - Sentence-level features

    Args:
        filepath: Path to merged_sentences.jsonl.gz
        sample_size: Number of tokens to sample

    Returns:
        DataFrame with token-level data and statistics dict
    """
    print(f"\n{'='*80}")
    print(f"LOADING TOKEN-LEVEL DATA")
    print(f"{'='*80}\n")
    print(f"Source: {filepath}")

    records = []
    sentences_processed = 0
    tokens_with_att_run1 = 0
    tokens_with_att_run3 = 0

    with gzip.open(filepath, 'rt') as f:
        for line in tqdm(f, desc="Processing sentences"):
            sentence = json.loads(line)
            sent_id = sentence.get('id')
            sent_text = sentence.get('text', '')

            # Get data for each run
            pair1_treated = sentence.get('pair1_treated_run1')
            pair2_treated = sentence.get('pair2_treated_run3')
            pair2_control = sentence.get('pair2_control_run4')

            if not pair1_treated or not pair2_control:
                continue

            tokens_treated_run1 = pair1_treated.get('tokens', [])
            tokens_control_run4 = pair2_control.get('tokens', [])

            # Reconstruct sentence text from tokens if not present
            if not sent_text and tokens_treated_run1:
                sent_text = ''.join([t.get('text', '') for t in tokens_treated_run1])

            # Verify matching token counts for run1
            if len(tokens_treated_run1) != len(tokens_control_run4):
                continue

            # Process sentence with spaCy for linguistic features
            spacy_doc = None
            if nlp is not None and sent_text:
                try:
                    spacy_doc = nlp(sent_text)
                except Exception as e:
                    # If spaCy processing fails, continue without linguistic features
                    pass

            # Compute sentence-level stats for run1
            sent_stats_run1 = compute_sentence_ll_stats(tokens_treated_run1)

            # Process each token for run1
            for token_idx in range(len(tokens_treated_run1)):
                treated_ll = tokens_treated_run1[token_idx].get('log_prob')
                control_ll = tokens_control_run4[token_idx].get('log_prob')

                if treated_ll is None or control_ll is None:
                    continue
                if np.isnan(treated_ll) or np.isnan(control_ll):
                    continue

                # Token-level ATT
                token_att = treated_ll - control_ll

                # Extract features
                features = extract_token_features(token_idx, tokens_treated_run1, sent_stats_run1, spacy_doc)

                # Create record
                record = {
                    'id': f"{sent_id}_run1_token{token_idx}",
                    'sentence_id': sent_id,
                    'sentence_text': sent_text,
                    'run': 'run1',
                    'att': token_att,
                    'treated_ll': treated_ll,
                    'control_ll': control_ll,
                    **features
                }

                records.append(record)
                tokens_with_att_run1 += 1

            # Process run3 if available
            if pair2_treated:
                tokens_treated_run3 = pair2_treated.get('tokens', [])

                if len(tokens_treated_run3) == len(tokens_control_run4):
                    sent_stats_run3 = compute_sentence_ll_stats(tokens_treated_run3)

                    for token_idx in range(len(tokens_treated_run3)):
                        treated_ll = tokens_treated_run3[token_idx].get('log_prob')
                        control_ll = tokens_control_run4[token_idx].get('log_prob')

                        if treated_ll is None or control_ll is None:
                            continue
                        if np.isnan(treated_ll) or np.isnan(control_ll):
                            continue

                        token_att = treated_ll - control_ll
                        features = extract_token_features(token_idx, tokens_treated_run3, sent_stats_run3, spacy_doc)

                        record = {
                            'id': f"{sent_id}_run3_token{token_idx}",
                            'sentence_id': sent_id,
                            'sentence_text': sent_text,
                            'run': 'run3',
                            'att': token_att,
                            'treated_ll': treated_ll,
                            'control_ll': control_ll,
                            **features
                        }

                        records.append(record)
                        tokens_with_att_run3 += 1

            sentences_processed += 1

    print(f"\n✓ Processed {sentences_processed:,} sentences")
    print(f"  Tokens with ATT (run1): {tokens_with_att_run1:,}")
    print(f"  Tokens with ATT (run3): {tokens_with_att_run3:,}")
    print(f"  Total tokens: {len(records):,}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Sample if too large
    if len(df) > sample_size:
        print(f"\nRandomly sampling {sample_size:,} tokens from {len(df):,} total")
        df = df.sample(n=sample_size, random_state=42)

    stats = {
        'sentences_processed': sentences_processed,
        'tokens_run1': tokens_with_att_run1,
        'tokens_run3': tokens_with_att_run3,
        'total_tokens': len(df)
    }

    return df, stats


def train_token_att_predictor(df: pd.DataFrame):
    """
    Train linear regression to predict token-level ATT.

    Args:
        df: DataFrame with token-level features and ATT values
    """
    print(f"\n{'='*80}")
    print(f"TRAINING TOKEN-LEVEL ATT PREDICTOR")
    print(f"{'='*80}\n")

    # ATT statistics
    print(f"Token-level ATT Statistics:")
    print(f"  Mean: {df['att'].mean():.6f}")
    print(f"  Median: {df['att'].median():.6f}")
    print(f"  Std: {df['att'].std():.6f}")
    print(f"  Min: {df['att'].min():.6f}")
    print(f"  Max: {df['att'].max():.6f}\n")

    # Select features (exclude metadata and categorical string features)
    exclude_cols = [
        'id', 'sentence_id', 'sentence_text', 'token_text', 'att',
        'treated_ll', 'control_ll', 'run',
        # Exclude categorical string features (will use boolean indicators instead)
        'pos', 'tag', 'dep', 'head_pos', 'head_dep', 'entity_type', 'entity_iob'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Using {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")
    print()

    X = df[feature_cols]
    y = df['att']

    # Handle any remaining NaN values
    print(f"Handling missing values...")
    nan_counts_before = X.isna().sum().sum()
    X = X.fillna(0)
    print(f"  Filled {nan_counts_before} NaN values with 0\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train):,} tokens")
    print(f"Test set: {len(X_test):,} tokens\n")

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
    print(f"TOP FEATURES PREDICTING TOKEN-LEVEL ATT")
    print(f"{'='*80}\n")

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_
    })

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

    # Save outputs
    output_file = Path("/tmp/token_att_feature_importance.csv")
    feature_importance.to_csv(output_file, index=False)
    print(f"\n✓ Saved feature importance to {output_file}")

    results_df = pd.DataFrame({
        'true_att': y_test,
        'predicted_att': y_test_pred,
        'residual': y_test - y_test_pred
    })
    results_file = Path("/tmp/token_att_predictions.csv")
    results_df.to_csv(results_file, index=False)
    print(f"✓ Saved predictions to {results_file}")

    return model, scaler, feature_cols, feature_importance


def main():
    """Main entry point."""
    merged_sentences_file = Path("/tmp/merged_sentences.jsonl.gz")

    if not merged_sentences_file.exists():
        print(f"Error: {merged_sentences_file} not found")
        print("Please run analyze_sentences.py first to generate merged token data")
        return

    # Load token-level data
    df, stats = load_token_level_data(merged_sentences_file, sample_size=100000)

    if len(df) == 0:
        print("Error: No valid token-level data found")
        return

    # Train predictor
    model, scaler, feature_cols, feature_importance = train_token_att_predictor(df)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
