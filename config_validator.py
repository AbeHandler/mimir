#!/usr/bin/env python3
"""
config_validator.py - Validate MIMIR config files

This script validates that config filenames match their base_model specification:
- If base_model contains "blocksbin" -> filename must contain ".blocks."
- If base_model contains "noblocksbin" -> filename must contain ".noblocks."
"""

import json
import sys
import argparse
from pathlib import Path


def validate_config(config_path: str) -> bool:
    """
    Validate that config filename matches base_model specification.

    Args:
        config_path: Path to the config file

    Returns:
        True if valid, False otherwise
    """
    config_file = Path(config_path)

    # Check if file exists
    if not config_file.exists():
        print(f"❌ Error: Config file does not exist: {config_path}")
        return False

    # Load config
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in config file: {e}")
        return False

    # Get base_model
    base_model = config.get("base_model")
    if not base_model:
        print(f"❌ Error: 'base_model' field not found in config")
        return False

    filename = config_file.name

    # Validate filename matches base_model
    if "noblocksbin" in base_model:
        if ".noblocks." not in filename:
            print(f"❌ Validation failed!")
            print(f"   Config: {filename}")
            print(f"   base_model: {base_model}")
            print(f"   Expected: Filename must contain '.noblocks.' for noblocksbin model")
            return False
    elif "blocksbin" in base_model:
        if ".blocks." not in filename:
            print(f"❌ Validation failed!")
            print(f"   Config: {filename}")
            print(f"   base_model: {base_model}")
            print(f"   Expected: Filename must contain '.blocks.' for blocksbin model")
            return False

    print(f"✅ Valid: {filename} matches base_model '{base_model}'")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate MIMIR config file naming conventions"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config file (e.g., configs/bothbins.blocks.lite.json)"
    )

    args = parser.parse_args()

    is_valid = validate_config(args.config)

    # Exit with appropriate status code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
