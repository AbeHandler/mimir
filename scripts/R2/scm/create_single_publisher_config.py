#!/usr/bin/env python3
"""
Create publisher-specific config files from hawaiinewsnow template.

Reads hawaiinewsnow_scm config templates and replaces publisher name.
"""
import argparse
import json
from pathlib import Path


NAME = "create_single_publisher_config"


def create_publisher_config(publisher: str, config_dir: Path) -> None:
    """
    Create config files for a publisher from hawaiinewsnow templates.

    Args:
        publisher: Publisher name to use in configs
        config_dir: Directory containing config files
    """
    variants = ["blocks", "noblocks"]

    for variant in variants:
        template_path = config_dir / f"hawaiinewsnow_scm.{variant}.lite.json"
        output_path = config_dir / f"{publisher}_scm.{variant}.lite.json"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Read template
        with open(template_path, 'r') as f:
            content = f.read()

        # Replace all instances of hawaiinewsnow with publisher
        updated_content = content.replace("hawaiinewsnow", publisher)

        # Validate JSON
        json.loads(updated_content)

        # Write output
        with open(output_path, 'w') as f:
            f.write(updated_content)

        print(f"Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create publisher-specific config files from hawaiinewsnow template"
    )
    parser.add_argument(
        "-publisher",
        type=str,
        help="Publisher name to use in config files"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing config files (default: configs)"
    )

    args = parser.parse_args()

    create_publisher_config(args.publisher, args.config_dir)


if __name__ == "__main__":
    main()
