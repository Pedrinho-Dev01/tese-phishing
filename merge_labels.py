"""
merge_labels.py
───────────────
Applies a principled emotion merge strategy based on co-occurrence
and phi coefficient analysis, then writes a new CSV ready for training.

Merge strategy (only positive phi pairs + semantic sense):
  excitement + surprise + joy + desire  → positive_arousal
  admiration + gratitude + caring       → warmth
  anger + fear                          → threat
  confusion                             → confusion  (kept)
  curiosity                             → curiosity  (kept)
  sadness                               → sadness    (kept)
  relief                                → relief     (kept)
  neutral + unsure                      → DROPPED    (too rare, low signal)

Result: 15 classes → 8 classes

Usage:
    python merge_labels.py

Outputs:
    code/combined_emails_dataset_merged.csv   ← use this for training
    merge_stats.txt                           ← summary of what changed
"""

import pandas as pd
import numpy as np
from collections import Counter

# ── Merge map ─────────────────────────────────────────────────────────────────
# Each original label maps to its new label, or None to drop it.
MERGE_MAP = {
    # Positive high-arousal cluster
    'excitement': 'positive_arousal',
    'surprise':   'positive_arousal',
    'joy':        'positive_arousal',
    'desire':     'positive_arousal',
    # Warmth / prosocial cluster
    'admiration': 'warmth',
    'gratitude':  'warmth',
    'caring':     'warmth',
    # Threat cluster
    'anger':      'threat',
    'fear':       'threat',
    # Kept as-is
    'confusion':  'confusion',
    'curiosity':  'curiosity',
    'sadness':    'sadness',
    'relief':     'relief',
    # Dropped — too rare and low signal
    'neutral':    None,
    'unsure':     None,
}

INPUT_CSV  = 'code/combined_emails_dataset.csv'
OUTPUT_CSV = 'code/combined_emails_dataset_merged.csv'
STATS_FILE = 'merge_stats.txt'

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df_phishing = df[df['label'] == 'phishing'].copy()
print(f"Total rows         : {len(df):,}")
print(f"Phishing rows      : {len(df_phishing):,}")

# Keep non-phishing rows untouched (no emotion column needed for them)
df_other = df[df['label'] != 'phishing'].copy()

# ── Apply merge to phishing rows ──────────────────────────────────────────────
def merge_emotion_list(emotion_str):
    """
    Input:  "curiosity#excitement#fear"
    Output: "curiosity#positive_arousal#threat"   (deduplicated, sorted)
            or NaN if all labels were dropped
    """
    if pd.isna(emotion_str):
        return np.nan
    original = [e.strip() for e in emotion_str.split('#') if e.strip()]
    merged = set()
    for e in original:
        new_label = MERGE_MAP.get(e)
        if new_label is not None:        # None means drop
            merged.add(new_label)
    if not merged:
        return np.nan
    return '#'.join(sorted(merged))

df_phishing['emotion_original'] = df_phishing['emotion']
df_phishing['emotion'] = df_phishing['emotion'].apply(merge_emotion_list)

# Drop rows where all emotions were dropped (all-None mappings)
before = len(df_phishing)
df_phishing = df_phishing.dropna(subset=['emotion'])
after = len(df_phishing)
print(f"\nPhishing rows after dropping all-None emotion rows: {after:,}  (dropped {before - after})")

# ── Stats ─────────────────────────────────────────────────────────────────────
new_emotion_lists = df_phishing['emotion'].str.split('#')
all_new_labels = [e for lst in new_emotion_lists for e in lst]
new_label_counts = Counter(all_new_labels)

new_label_matrix_rows = []
new_classes = sorted(new_label_counts.keys())
for lst in new_emotion_lists:
    row = {c: 0 for c in new_classes}
    for e in lst:
        row[e] = 1
    new_label_matrix_rows.append(row)
new_label_df = pd.DataFrame(new_label_matrix_rows)
labels_per_sample = new_label_df.sum(axis=1)

stats_lines = []
stats_lines.append("=" * 60)
stats_lines.append("MERGED LABEL DISTRIBUTION")
stats_lines.append("=" * 60)
for label in sorted(new_label_counts, key=lambda x: -new_label_counts[x]):
    count = new_label_counts[label]
    pct   = count / after * 100
    bar   = "█" * int(pct / 2)
    stats_lines.append(f"  {label:<25} {count:>5,}  ({pct:5.1f}%)  {bar}")

stats_lines.append("")
stats_lines.append("=" * 60)
stats_lines.append("LABELS PER SAMPLE (after merge)")
stats_lines.append("=" * 60)
for n, cnt in sorted(Counter(labels_per_sample.astype(int)).items()):
    bar = "█" * int(cnt / after * 50)
    stats_lines.append(f"  {n} label(s): {cnt:>5,}  ({cnt/after*100:5.1f}%)  {bar}")
stats_lines.append(f"\n  Mean  : {labels_per_sample.mean():.3f}")
stats_lines.append(f"  Median: {labels_per_sample.median():.1f}")

stats_lines.append("")
stats_lines.append("=" * 60)
stats_lines.append("MERGE MAP APPLIED")
stats_lines.append("=" * 60)
grouped = {}
for orig, new in MERGE_MAP.items():
    grouped.setdefault(new or 'DROPPED', []).append(orig)
for new_label, originals in sorted(grouped.items()):
    stats_lines.append(f"  {new_label:<25} ← {', '.join(sorted(originals))}")

stats_text = "\n".join(stats_lines)
print("\n" + stats_text)

with open(STATS_FILE, 'w') as f:
    f.write(stats_text + "\n")
print(f"\nStats written to {STATS_FILE}")

# ── Save ──────────────────────────────────────────────────────────────────────
# Recombine with non-phishing rows and save
df_out = pd.concat([df_phishing.drop(columns=['emotion_original']), df_other], ignore_index=True)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Merged dataset written to {OUTPUT_CSV}")
print(f"  Total rows : {len(df_out):,}")
print(f"  Phishing   : {len(df_phishing):,}")
print(f"  Other      : {len(df_other):,}")
print(f"\nUse '{OUTPUT_CSV}' as the input CSV in your training scripts.")
print("Remember to update the csv path at the top of each training script.")