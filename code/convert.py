import csv
import json
import random
import sys
from pathlib import Path
from collections import Counter


def random_hex_color() -> str:
    return "#{:06X}".format(random.randint(0, 0xFFFFFF))


def contrasting_text_color(bg_hex: str) -> str:
    """Return black or white depending on background luminance."""
    bg_hex = bg_hex.lstrip("#")
    r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


def build_label_entries(unique_labels: list[str]) -> list[dict]:
    entries = []
    used_keys = set()

    for label in unique_labels:
        # Pick a suffix_key: try first letter, then subsequent letters, then fallback
        suffix_key = None
        for ch in label.lower():
            if ch.isalpha() and ch not in used_keys:
                suffix_key = ch
                break
        if suffix_key is None:
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch not in used_keys:
                    suffix_key = ch
                    break
        used_keys.add(suffix_key)

        bg = random_hex_color()
        entries.append({
            "text": label,
            "suffix_key": suffix_key,
            "background_color": bg,
            "text_color": contrasting_text_color(bg),
        })

    return entries


def extract_labels_to_file(input_filepath: str, output_filepath: str = "labels.json") -> str:
    path = Path(input_filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_filepath}")

    all_labels = []

    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for record in data:
            label_field = record.get("label", "")
            if label_field:
                all_labels.extend([l.strip() for l in label_field.split("#") if l.strip()])

    elif path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            for record in csv.DictReader(f):
                label_field = record.get("label", "")
                if label_field:
                    all_labels.extend([l.strip() for l in label_field.split("#") if l.strip()])

    else:
        raise ValueError("Unsupported file type. Use .json or .csv")

    unique_labels = sorted(Counter(all_labels).keys())
    entries = build_label_entries(unique_labels)

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    print(f"✅ Extracted {len(entries)} labels → '{output_filepath}'")
    return output_filepath


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solution.py <input.json|input.csv> [output.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "labels.json"

    extract_labels_to_file(input_file, output_file)