"""
Reconstructor - Rebuild documents from MIMIR results with per-token scores

This module provides functionality to reconstruct documents from MIMIR output files,
aligning tokens with their log probability scores.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
import argparse
import spacy
from tqdm import tqdm
from sentence_splitter import SentenceSplitter


@dataclass
class Token:
    """A single token with its text and log probability score."""
    text: str
    log_prob: float
    token_id: int

    def __repr__(self):
        return f"Token(text={repr(self.text)}, log_prob={self.log_prob:.4f})"


@dataclass
class Sentence:
    """A sentence with its tokens and scores."""
    text: str
    tokens: List[Token]
    start_idx: int  # Token index where sentence starts in document
    end_idx: int    # Token index where sentence ends in document

    def __repr__(self):
        return f"Sentence(text={self.text[:50]}..., n_tokens={len(self.tokens)})"


@dataclass
class Document:
    """A document reconstructed with tokens and their scores."""
    doc_id: str
    full_text: str
    tokens: List[Token]
    sentences: List[Sentence]
    membership: str  # 'member' or 'nonmember'
    aggregated_score: float

    def __repr__(self):
        return f"Document(id={self.doc_id}, membership={self.membership}, n_tokens={len(self.tokens)}, n_sentences={len(self.sentences)})"

    def get_token_texts(self) -> List[str]:
        """Get list of token texts."""
        return [token.text for token in self.tokens]

    def get_log_probs(self) -> List[float]:
        """Get list of log probabilities."""
        return [token.log_prob for token in self.tokens]


class Reconstructor:
    """
    Reconstructs documents from MIMIR results JSON files.

    Usage:
        reconstructor = Reconstructor("path/to/loss_results.json")
        documents = reconstructor.reconstruct_all()

        # Access specific document
        doc = documents[0]
        print(doc.full_text)
        print(doc.tokens)
        print(doc.sentences)
    """

    def __init__(self, results_json_path: str, spacy_model: str = "en_core_web_sm"):
        """
        Initialize reconstructor with path to results JSON file.

        Args:
            results_json_path: Path to *_results.json file from MIMIR
            spacy_model: Spacy model to use for sentence splitting (default: en_core_web_sm)
        """
        self.results_path = Path(results_json_path)

        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_json_path}")

        # Load results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)

        # Load config from same directory
        config_path = self.results_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load tokenizer
        model_name = self.config.get('base_model')
        if not model_name:
            raise ValueError("base_model not found in config")

        print(f"Loading tokenizer for model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load spacy for sentence splitting
        print(f"Loading spacy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Spacy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")
            raise

        # Extract attack name from filename
        self.attack_name = self.results_path.stem.replace('_results', '')

    def _split_into_sentences(self, full_text: str, tokens: List[Token]) -> List[Sentence]:
        """
        Split document into sentences using spacy, aligning with tokens.

        Args:
            full_text: Full document text
            tokens: List of Token objects (note: first token has no score due to shift)

        Returns:
            List of Sentence objects
        """
        # Use spacy to split into sentences
        doc = self.nlp(full_text)
        sentences = []

        # Reconstruct full text from tokens to map character positions
        # Note: Token 0 has no log prob (it's skipped due to shift in MIMIR)
        reconstructed_text = ""
        char_to_token_idx = []

        for i, token in enumerate(tokens):
            start_char = len(reconstructed_text)
            reconstructed_text += token.text
            end_char = len(reconstructed_text)
            # Map each character to its token index
            char_to_token_idx.extend([i] * (end_char - start_char))

        # For each spacy sentence, find the corresponding tokens
        for sent in doc.sents:
            sent_text = sent.text
            sent_start_char = sent.start_char
            sent_end_char = sent.end_char

            # Find token indices for this sentence
            # Handle edge cases where character mapping might be off
            if sent_start_char < len(char_to_token_idx) and sent_end_char <= len(char_to_token_idx):
                start_token_idx = char_to_token_idx[sent_start_char] if sent_start_char < len(char_to_token_idx) else 0
                end_token_idx = char_to_token_idx[sent_end_char - 1] + 1 if sent_end_char > 0 and sent_end_char - 1 < len(char_to_token_idx) else len(tokens)
            else:
                # Fallback: approximate based on position
                continue

            sent_tokens = tokens[start_token_idx:end_token_idx]

            if sent_tokens:  # Only add if we have tokens
                sentences.append(Sentence(
                    text=sent_text,
                    tokens=sent_tokens,
                    start_idx=start_token_idx,
                    end_idx=end_token_idx
                ))

        return sentences

    def reconstruct_document(
        self,
        doc_id: str,
        membership: str
    ) -> Optional[Document]:
        """
        Reconstruct a single document with aligned tokens and scores.

        Args:
            doc_id: Document ID
            membership: 'member' or 'nonmember'

        Returns:
            Document object with tokens and scores, or None if not found
        """
        # Get data for this document
        id_to_text = self.results.get('id_to_text', {}).get(membership, {})
        id_to_token_probs = self.results.get('id_to_token_probs', {}).get(membership, {})
        id_to_score = self.results.get('id_to_score', {}).get(membership, {})

        if doc_id not in id_to_text:
            return None

        # Get text and token probs
        text_list = id_to_text[doc_id]
        token_probs_list = id_to_token_probs[doc_id]
        aggregated_score = id_to_score.get(doc_id, float('nan'))

        # Text and token_probs are stored as lists (for full_doc mode)
        # Since full_doc=False in current setup, they have 1 element
        if isinstance(text_list, list) and len(text_list) > 0:
            full_text = text_list[0]
            token_log_probs = token_probs_list[0]
        else:
            full_text = text_list
            token_log_probs = token_probs_list

        # Tokenize the text
        token_ids = self.tokenizer.encode(full_text)

        # MIMIR skips the first token due to shift in language modeling
        # (predicting token N from tokens 0..N-1)
        # So we expect: len(token_ids) = len(token_log_probs) + 1
        expected_probs = len(token_ids) - 1

        if len(token_log_probs) != expected_probs:
            print(f"Warning: Unexpected token count for {doc_id}")
            print(f"  Tokenizer produced {len(token_ids)} tokens")
            print(f"  Expected {expected_probs} log probs (N-1 due to shift)")
            print(f"  Got {len(token_log_probs)} log probs")
            # Use minimum to avoid index errors
            min_len = min(len(token_ids) - 1, len(token_log_probs))
            token_log_probs = token_log_probs[:min_len]
            token_ids = token_ids[:min_len + 1]

        # Create Token objects
        # token_ids[0] doesn't have a log prob (it's the conditioning token)
        # token_ids[1] has log_prob[0], token_ids[2] has log_prob[1], etc.
        tokens = []

        # First token has no log probability (it's what we condition on)
        first_token_text = self.tokenizer.decode([token_ids[0]])
        tokens.append(Token(
            text=first_token_text,
            log_prob=float('nan'),  # No log prob for first token
            token_id=token_ids[0]
        ))

        # Remaining tokens have log probs
        for i, (token_id, log_prob) in enumerate(zip(token_ids[1:], token_log_probs)):
            token_text = self.tokenizer.decode([token_id])
            tokens.append(Token(
                text=token_text,
                log_prob=log_prob,
                token_id=token_id
            ))

        # Split into sentences
        sentences = self._split_into_sentences(full_text, tokens)

        return Document(
            doc_id=doc_id,
            full_text=full_text,
            tokens=tokens,
            sentences=sentences,
            membership=membership,
            aggregated_score=aggregated_score
        )

    def reconstruct_all(self) -> List[Document]:
        """
        Reconstruct all documents from the results file.

        Returns:
            List of Document objects
        """
        documents = []

        for membership in ['member', 'nonmember']:
            id_to_text = self.results.get('id_to_text', {}).get(membership, {})

            for doc_id in tqdm(id_to_text.keys(), desc=f"Reconstructing {membership}s"):
                doc = self.reconstruct_document(doc_id, membership)
                if doc:
                    documents.append(doc)

        print(f"\nReconstructed {len(documents)} documents")
        return documents

    def reconstruct_members_only(self) -> List[Document]:
        """
        Reconstruct only member documents from the results file.

        Returns:
            List of Document objects (members only)
        """
        documents = []
        id_to_text = self.results.get('id_to_text', {}).get('member', {})

        for doc_id in tqdm(id_to_text.keys(), desc=f"Reconstructing members"):
            doc = self.reconstruct_document(doc_id, 'member')
            if doc:
                documents.append(doc)

        print(f"\nReconstructed {len(documents)} member documents")
        return documents

    def get_document_ids(self, membership: Optional[str] = None) -> List[str]:
        """
        Get list of document IDs.

        Args:
            membership: Filter by 'member' or 'nonmember', or None for all

        Returns:
            List of document IDs
        """
        ids = []

        memberships = [membership] if membership else ['member', 'nonmember']

        for m in memberships:
            id_to_text = self.results.get('id_to_text', {}).get(m, {})
            ids.extend(id_to_text.keys())

        return ids

    def export_sentences_to_json(self, output_root: str = None, batch_size: int = 50) -> str:
        """
        Export all documents split into sentences as JSONL file.

        Each sentence is exported with:
        - ID (format: d{doc_idx}s{sent_idx})
        - text
        - tokens (list of dicts with text, log_prob, token_id)

        Args:
            output_root: Root directory for output. If None, uses /tmp/
                        The full path structure from results file is preserved under this root.
            batch_size: Batch size for spacy processing (default: 50)

        Returns:
            Path to output directory
        """
        # Determine output directory by preserving path structure
        if output_root is None:
            output_root = "/tmp"

        # Get the full directory path of the results file relative to current working directory
        # e.g., tmp_results/.../loss_results.json -> tmp_results/.../
        results_dir = self.results_path.parent

        # Combine output_root with the results directory structure
        output_path = Path(output_root) / results_dir
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting sentences to: {output_path}")

        # Reconstruct only member documents
        documents = self.reconstruct_members_only()

        # Initialize sentence splitter
        splitter = SentenceSplitter()

        # Batch process all documents for sentence splitting
        print(f"Splitting {len(documents)} documents into sentences...")
        all_texts = [doc.full_text for doc in documents]
        all_sentence_lists = splitter.split_batch(all_texts, batch_size=batch_size)

        # Export sentences
        all_sentences_data = []

        for doc_idx, (doc, sentence_texts) in enumerate(tqdm(
            zip(documents, all_sentence_lists),
            total=len(documents),
            desc="Processing sentences"
        )):

            # For each sentence, find corresponding tokens
            char_offset = 0
            for sent_idx, sent_text in enumerate(sentence_texts):
                # Find start and end char positions
                sent_start = doc.full_text.find(sent_text, char_offset)
                if sent_start == -1:
                    continue
                sent_end = sent_start + len(sent_text)
                char_offset = sent_end

                # Map character positions to token indices
                # Reconstruct character mapping from tokens
                reconstructed_text = ""
                char_to_token_idx = []
                for tok_idx, token in enumerate(doc.tokens):
                    start_char = len(reconstructed_text)
                    reconstructed_text += token.text
                    end_char = len(reconstructed_text)
                    char_to_token_idx.extend([tok_idx] * (end_char - start_char))

                # Find tokens for this sentence
                if sent_start < len(char_to_token_idx) and sent_end <= len(char_to_token_idx):
                    start_token_idx = char_to_token_idx[sent_start] if sent_start < len(char_to_token_idx) else 0
                    end_token_idx = char_to_token_idx[sent_end - 1] + 1 if sent_end > 0 and sent_end - 1 < len(char_to_token_idx) else len(doc.tokens)
                else:
                    # Skip if character mapping is off
                    continue

                sent_tokens = doc.tokens[start_token_idx:end_token_idx]

                # Create sentence data
                sentence_data = {
                    "id": f"d{doc_idx}s{sent_idx}",
                    "doc_id": doc.doc_id,
                    "membership": doc.membership,
                    "text": sent_text,
                    "tokens": [
                        {
                            "text": tok.text,
                            "log_prob": tok.log_prob,
                            "token_id": tok.token_id
                        }
                        for tok in sent_tokens
                    ]
                }

                all_sentences_data.append(sentence_data)

        # Write to JSONL file (one JSON object per line)
        output_file = output_path / "sentences.jsonl"
        with open(output_file, 'w') as f:
            for sentence_data in all_sentences_data:
                f.write(json.dumps(sentence_data) + '\n')

        print(f"\nExported {len(all_sentences_data)} sentences to {output_file}")

        return str(output_path)


def main():
    """Example usage with argparse."""
    parser = argparse.ArgumentParser(
        description='Reconstruct documents from MIMIR results JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reconstructor.py --results-json tmp_results/.../loss_results.json
  python reconstructor.py --results-json tmp_results/.../loss_results.json --spacy-model en_core_web_lg
        """
    )
    parser.add_argument(
        '--results-json',
        type=str,
        required=True,
        help='Path to *_results.json file from MIMIR'
    )
    parser.add_argument(
        '--spacy-model',
        type=str,
        default='en_core_web_sm',
        help='Spacy model for sentence splitting (default: en_core_web_sm)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default=None,
        help='Root directory for output. Path structure from results file is preserved under this root. (default: /tmp/)'
    )

    args = parser.parse_args()

    # Create reconstructor
    # Note: argparse converts --results-json to args.results_json (underscores)
    reconstructor = Reconstructor(args.results_json, spacy_model=args.spacy_model)

    # Show member document IDs
    print("\n=== Member Document IDs ===")
    member_ids = reconstructor.get_document_ids('member')
    print(f"Members: {len(member_ids)}")

    # Export sentences (always)
    print("\n=== Exporting Sentences ===")
    output_dir = reconstructor.export_sentences_to_json(output_root=args.output_root)
    print(f"\n✓ Sentences exported to: {output_dir}")


if __name__ == "__main__":
    main()
