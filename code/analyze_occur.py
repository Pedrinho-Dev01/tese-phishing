"""
analyze_label_cooccurrence.py
─────────────────────────────
Checks the multi-label structure of your emotion dataset:
  1. Label counts and single-vs-multi-label split
  2. Average labels per sample
  3. Co-occurrence matrix (which emotions appear together)
  4. Correlation between label pairs (phi coefficient)
  5. Suggested merges based on high correlation

Run from the repo root:
    python analyze_label_cooccurrence.py
"""

import pandas as pd
import numpy as np

# ── Load & parse ──────────────────────────────────────────────────────────────
df = pd.read_csv('code/combined_emails_dataset.csv')
df = df[df['label'] == 'phishing'].copy()
df = df.dropna(subset=['emotion'])
df['emotion_list'] = df['emotion'].str.split('#')

emotion_classes = sorted(set(e for emotions in df['emotion_list'] for e in emotions))
label2id = {e: i for i, e in enumerate(emotion_classes)}
num_labels = len(emotion_classes)

def encode_labels(emotion_list):
    vec = np.zeros(num_labels, dtype=np.float32)
    for e in emotion_list:
        if e in label2id:
            vec[label2id[e]] = 1.0
    return vec

label_matrix = np.stack(df['emotion_list'].apply(encode_labels).values)
N = len(label_matrix)

# ── 1. Label counts ───────────────────────────────────────────────────────────
print("=" * 65)
print("1. LABEL COUNTS")
print("=" * 65)
counts = label_matrix.sum(axis=0).astype(int)
for emotion, count in sorted(zip(emotion_classes, counts), key=lambda x: -x[1]):
    bar = "█" * int(count / N * 50)
    print(f"  {emotion:<25} {count:>5,}  ({count/N*100:5.1f}%)  {bar}")

# ── 2. Labels per sample distribution ────────────────────────────────────────
print("\n" + "=" * 65)
print("2. LABELS PER SAMPLE")
print("=" * 65)
labels_per_sample = label_matrix.sum(axis=1).astype(int)
value_counts = pd.Series(labels_per_sample).value_counts().sort_index()
for n_labels, count in value_counts.items():
    bar = "█" * int(count / N * 50)
    print(f"  {n_labels} label(s): {count:>5,}  ({count/N*100:5.1f}%)  {bar}")

print(f"\n  Mean labels per sample : {labels_per_sample.mean():.3f}")
print(f"  Median                 : {np.median(labels_per_sample):.1f}")
print(f"  Samples with 1 label   : {(labels_per_sample == 1).sum():,}  ({(labels_per_sample == 1).mean()*100:.1f}%)")
print(f"  Samples with >1 labels : {(labels_per_sample > 1).sum():,}  ({(labels_per_sample > 1).mean()*100:.1f}%)")
print(f"  Samples with 0 labels  : {(labels_per_sample == 0).sum():,}  ← should be 0")

# ── 3. Co-occurrence matrix ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. CO-OCCURRENCE MATRIX (count of samples sharing both labels)")
print("=" * 65)

cooc = (label_matrix.T @ label_matrix).astype(int)
np.fill_diagonal(cooc, 0)

# Print only pairs with cooc > 0, sorted by frequency
pairs = []
for i in range(num_labels):
    for j in range(i + 1, num_labels):
        if cooc[i, j] > 0:
            pairs.append((emotion_classes[i], emotion_classes[j], cooc[i, j]))

pairs.sort(key=lambda x: -x[2])
if pairs:
    print(f"\n  {'Emotion A':<25} {'Emotion B':<25} {'Co-occur':>8}  {'% of A':>8}  {'% of B':>8}")
    print("  " + "-" * 75)
    for a, b, c in pairs:
        pct_a = c / counts[label2id[a]] * 100
        pct_b = c / counts[label2id[b]] * 100
        print(f"  {a:<25} {b:<25} {c:>8,}  {pct_a:>7.1f}%  {pct_b:>7.1f}%")
else:
    print("\n  No co-occurrences found — all samples have exactly 1 label.")
    print("  This is the most likely cause of poor multi-label performance.")

# ── 4. Phi coefficient (correlation) between label pairs ─────────────────────
print("\n" + "=" * 65)
print("4. PHI COEFFICIENT — label-pair correlation (|phi| > 0.3 = notable)")
print("=" * 65)

phi_pairs = []
for i in range(num_labels):
    for j in range(i + 1, num_labels):
        a = label_matrix[:, i]
        b = label_matrix[:, j]
        n11 = (a * b).sum()
        n10 = (a * (1 - b)).sum()
        n01 = ((1 - a) * b).sum()
        n00 = ((1 - a) * (1 - b)).sum()
        denom = np.sqrt((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01))
        phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0
        phi_pairs.append((emotion_classes[i], emotion_classes[j], float(phi)))

phi_pairs.sort(key=lambda x: -abs(x[2]))

notable = [(a, b, phi) for a, b, phi in phi_pairs if abs(phi) > 0.1]
if notable:
    print(f"\n  {'Emotion A':<25} {'Emotion B':<25} {'Phi':>8}  Interpretation")
    print("  " + "-" * 75)
    for a, b, phi in notable:
        if abs(phi) >= 0.3:
            interp = "★ STRONG — consider merging"
        elif abs(phi) >= 0.2:
            interp = "moderate"
        else:
            interp = "weak"
        sign = "+" if phi > 0 else "-"
        print(f"  {a:<25} {b:<25} {phi:>+8.3f}  {interp}")
else:
    print("\n  No notable correlations found (all |phi| < 0.1).")

# ── 5. Merge suggestions ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("5. MERGE SUGGESTIONS (phi >= 0.30)")
print("=" * 65)

strong = [(a, b, phi) for a, b, phi in phi_pairs if phi >= 0.30]
if strong:
    print("\n  These pairs always or almost always co-occur.")
    print("  Merging them reduces label count and improves learnability.\n")
    for a, b, phi in strong:
        print(f"  → Merge '{a}' + '{b}'  (phi={phi:.3f})")
else:
    print("\n  No strongly correlated pairs found.")
    print("  If most samples have 1 label, the problem is data sparsity,")
    print("  not label overlap — focus on generating more multi-label samples.")

# ── Summary diagnosis ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("DIAGNOSIS SUMMARY")
print("=" * 65)
single_label_pct = (labels_per_sample == 1).mean() * 100
multi_label_pct  = (labels_per_sample > 1).mean() * 100

print(f"\n  Dataset size       : {N:,} phishing emails")
print(f"  Emotion classes    : {num_labels}")
print(f"  Single-label       : {single_label_pct:.1f}%")
print(f"  Multi-label        : {multi_label_pct:.1f}%")
print(f"  Avg labels/sample  : {labels_per_sample.mean():.2f}")
print(f"  Co-occurring pairs : {len(pairs)}")

print()
if single_label_pct > 80:
    print("  ⚠  >80% of samples have only 1 label.")
    print("     The model has almost no examples of real co-occurrence.")
    print("     Recommendation: generate or annotate more multi-label samples,")
    print("     OR reframe as a single-label problem and accept the constraint.")
elif multi_label_pct > 30 and len(strong) > 0:
    print("  ℹ  Multi-label data exists but some labels are highly correlated.")
    print("     Consider the merges listed above.")
else:
    print("  ✓  Multi-label structure looks reasonable.")
    print("     Poor performance is likely due to model capacity or data size.")

print()