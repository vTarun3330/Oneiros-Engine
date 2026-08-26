"""Backward-compatible entry point for canonical Oneiros dataset preparation.

Use this instead of splitting raw mutation pairs directly.  The preparation
step removes invalid/duplicate/unexposed mutants, normalizes assertions, writes
the split manifest, and keeps all variants of a golden function in one split.
"""
from prepare_training_dataset import main


if __name__ == "__main__":
    main()
