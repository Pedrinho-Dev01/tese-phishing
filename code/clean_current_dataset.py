#!/usr/bin/env python3
"""Clean current dataset in place using existing preprocessing utilities."""

from pathlib import Path

import pandas as pd

from dataset_processor import (
    strip_trailing_artifacts,
    replace_placeholders,
    remove_prompt_type_column,
)


def _count_changes(before: pd.Series, after: pd.Series) -> int:
    """Count changed rows while treating NaN==NaN as unchanged."""
    marker = "__MISSING_VALUE__"
    return int((before.fillna(marker) != after.fillna(marker)).sum())


def clean_current_dataset_in_place(file_path: str = "code/current_dataset.csv") -> None:
    """Load, clean, and overwrite the current dataset CSV."""
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError(
            f"Missing required column 'text'. Available columns: {list(df.columns)}"
        )

    original_text = df["text"].copy()

    print("Cleaning trailing text artifacts...")
    text_after_artifacts = original_text.apply(strip_trailing_artifacts)
    artifacts_changed = _count_changes(original_text, text_after_artifacts)

    print("Replacing placeholders...")
    text_after_placeholders = text_after_artifacts.apply(replace_placeholders)
    placeholders_changed = _count_changes(text_after_artifacts, text_after_placeholders)

    df["text"] = text_after_placeholders

    print("Removing 'prompt_type' if present...")
    df = remove_prompt_type_column(df, "Current dataset")

    df.to_csv(csv_path, index=False)

    total_changed = _count_changes(original_text, df["text"])
    print("\nDone.")
    print(f"  Rows: {len(df):,}")
    print(f"  Text changed by artifact cleanup: {artifacts_changed:,}")
    print(f"  Text changed by placeholder replacement: {placeholders_changed:,}")
    print(f"  Total text rows changed: {total_changed:,}")
    print(f"  Saved in place to: {csv_path}")


if __name__ == "__main__":
    clean_current_dataset_in_place()
