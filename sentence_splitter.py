"""
Simple sentence splitter using spacy

This module provides a lightweight wrapper around spacy for sentence splitting.
"""

import spacy
from typing import List


class SentenceSplitter:
    """
    Simple sentence splitter using spacy.

    Usage:
        splitter = SentenceSplitter()
        sentences = splitter.split("Hello world. How are you?")
        # Returns: ["Hello world.", "How are you?"]
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize sentence splitter with spacy model.

        Args:
            model: Spacy model name (default: en_core_web_sm)
        """
        print(f"Loading spacy model: {model}")
        try:
            # Disable unnecessary components for speed
            self.nlp = spacy.load(model, disable=["tagger", "parser", "ner", "lemmatizer"])
            # Re-enable only sentencizer
            self.nlp.enable_pipe("sentencizer")
        except OSError:
            print(f"Spacy model '{model}' not found. Installing sentencizer...")
            # If model not found, use blank model with sentencizer
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        except ValueError:
            # If sentencizer doesn't exist in the loaded model, add it
            print("Adding sentencizer component...")
            self.nlp.add_pipe("sentencizer")

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text to split

        Returns:
            List of sentence strings
        """
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def split_batch(self, texts: List[str], batch_size: int = 50) -> List[List[str]]:
        """
        Split multiple texts into sentences using nlp.pipe for efficiency.

        Args:
            texts: List of texts to split
            batch_size: Batch size for processing (default: 50)

        Returns:
            List of lists of sentence strings (one list per input text)
        """
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            results.append([sent.text for sent in doc.sents])
        return results

    def split_with_offsets(self, text: str) -> List[tuple]:
        """
        Split text into sentences with character offsets.

        Args:
            text: Input text to split

        Returns:
            List of tuples (sentence_text, start_char, end_char)
        """
        doc = self.nlp(text)
        return [(sent.text, sent.start_char, sent.end_char) for sent in doc.sents]

    def split_batch_with_offsets(self, texts: List[str], batch_size: int = 50) -> List[List[tuple]]:
        """
        Split multiple texts into sentences with offsets using nlp.pipe for efficiency.

        Args:
            texts: List of texts to split
            batch_size: Batch size for processing (default: 50)

        Returns:
            List of lists of tuples (one list per input text)
            Each tuple is (sentence_text, start_char, end_char)
        """
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            results.append([(sent.text, sent.start_char, sent.end_char) for sent in doc.sents])
        return results


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Simple sentence splitter using spacy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentence_splitter.py --text "Hello world. How are you? I am fine."
  python sentence_splitter.py --text "First sentence. Second sentence." --model en_core_web_sm
        """
    )
    parser.add_argument(
        '--text',
        type=str,
        required=True,
        help='Text to split into sentences'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='en_core_web_sm',
        help='Spacy model to use (default: en_core_web_sm)'
    )

    args = parser.parse_args()

    # Create splitter
    splitter = SentenceSplitter(model=args.model)

    # Split text
    print("\n=== Input Text ===")
    print(args.text)

    print("\n=== Sentences ===")
    sentences = splitter.split(args.text)
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")

    print("\n=== Sentences with Offsets ===")
    sentences_with_offsets = splitter.split_with_offsets(args.text)
    for i, (sent, start, end) in enumerate(sentences_with_offsets, 1):
        print(f"{i}. [{start}:{end}] {sent}")


if __name__ == "__main__":
    main()
