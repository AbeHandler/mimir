#!/usr/bin/env python3
"""
Get the filesystem path where a model was saved based on config.

Given a config file, extracts the model path on the filesystem where
the trained model was saved according to the naming convention:
{BASE_MODEL_NAME}_{DATASET_NAME}{DEBUG_SUFFIX}
"""

import argparse
import json
import os
from pathlib import Path

NAME = "get_model_path"


def get_model_path_from_config(config_path: str) -> str:
    """
    Extract the model filesystem path from a config file.

    Args:
        config_path: Path to the JSON config file

    Returns:
        The directory path where the model is saved

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required fields are missing from config
        json.JSONDecodeError: If config is not valid JSON
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract base_model from config
    base_model = config.get('base_model')
    if not base_model:
        raise KeyError("Config missing required field: 'base_model'")

    # The base_model in the config is typically the HuggingFace repo of the trained model
    # e.g., "abehandlerorg/gpt-oss-20b_cptllama-2024-01-29-Y0_debug"
    # The local filesystem path is the part after the last "/"
    model_name = base_model.split('/')[-1]

    return model_name


def main():
    parser = argparse.ArgumentParser(
        description='Get filesystem path where a model was saved from config file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/get_model_path.py configs/gptosstmpcheck.debug.lite.json
    python scripts/get_model_path.py --check configs/gptosstmpcheck.debug.lite.json
    python scripts/get_model_path.py --base-dir /custom/path configs/gptosstmpcheck.debug.lite.json
    python scripts/get_model_path.py --check --absolute configs/gptosstmpcheck.debug.lite.json
        """
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to the JSON config file'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if the model directory exists on filesystem'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/home/abe/dolma/scripts/R2/create/cpt/training/',
        help='Base directory where models are saved (default: /home/abe/dolma/scripts/R2/create/cpt/training/)'
    )
    parser.add_argument(
        '--absolute',
        action='store_true',
        help='Return absolute path'
    )

    args = parser.parse_args()

    try:
        model_name = get_model_path_from_config(args.config_path)

        # Construct full path with base directory
        base_dir = Path(args.base_dir)
        model_path = base_dir / model_name

        if args.absolute:
            model_path = model_path.resolve()

        if args.check:
            exists = model_path.exists()
            status = "EXISTS" if exists else "NOT FOUND"
            print(f"{model_path} [{status}]")

            # Check for model completion indicator
            if exists:
                index_file = model_path / "model.safetensors.index.json"
                if index_file.exists():
                    print(f"  ✓ Model appears complete (found {index_file.name})")
                else:
                    print(f"  ⚠ Model may be incomplete (missing {index_file.name})")
        else:
            print(model_path)

    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
