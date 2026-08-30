"""Training CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_config
from src.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Embedding finetune training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
