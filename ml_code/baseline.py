"""
TF-IDF Baseline for Phishing Detection
=======================================
Mirrors the exact data pipeline from ml_code/roberta.py and ml_code/electra.py:
  - Same CSV source
  - Same label mapping (non-phishing=0, phishing=1)
  - Same stratified 80/20 train/test split (random_state=42)
  - Same 10% validation split from train (random_state=42)
  - Same metrics: accuracy, F1, precision, recall
  - Same threshold evaluation (0.5 standard + 0.35 custom)

Run with:
    python tfidf_baseline.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import issparse

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("=" * 60)
print("TF-IDF BASELINE — PHISHING DETECTION")
print("=" * 60)

df = pd.read_csv("code/combined_emails_dataset.csv")

# ── 2. Label mapping (identical to roberta.py / electra.py) ──────────────────
df["labels"] = df["label"].map({"non-phishing": 0, "phishing": 1})

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
print(df["labels"].value_counts())
print(f"Class 0 (ham):     {(df['labels']==0).sum()} ({(df['labels']==0).mean()*100:.1f}%)")
print(f"Class 1 (phishing):{(df['labels']==1).sum()} ({(df['labels']==1).mean()*100:.1f}%)")
print("=" * 60)

# ── 3. Splits — identical to transformer scripts ─────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["labels"], random_state=42
)
train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["labels"], random_state=42
)

print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

X_train = train_df["text"].fillna("").values
y_train = train_df["labels"].values
X_val   = val_df["text"].fillna("").values
y_val   = val_df["labels"].values
X_test  = test_df["text"].fillna("").values
y_test  = test_df["labels"].values

# ── 4. Class weights (same strategy as transformer scripts) ───────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
cw_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {cw_dict}")


# ── 5. Helper: evaluate at two thresholds ────────────────────────────────────
def evaluate(model_name: str, y_true: np.ndarray, proba: np.ndarray) -> dict:
    results = {}
    for threshold in [0.5, 0.35]:
        y_pred = (proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)
        label = f"Threshold {threshold}"
        results[threshold] = {
            "accuracy":  acc,
            "f1":        f1,
            "precision": precision,
            "recall":    recall,
        }
        print(f"\n  [{model_name}] {label}")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")

    try:
        auc_roc = roc_auc_score(y_true, proba)
        auc_pr  = average_precision_score(y_true, proba)
        print(f"\n  [{model_name}] ROC-AUC: {auc_roc:.4f}  |  PR-AUC: {auc_pr:.4f}")
        results["roc_auc"] = auc_roc
        results["pr_auc"]  = auc_pr
    except Exception:
        pass

    return results


# ── 6. Model A: TF-IDF + Logistic Regression ─────────────────────────────────
print("\n" + "=" * 60)
print("MODEL A: TF-IDF + Logistic Regression")
print("=" * 60)

lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        sublinear_tf=True,      # apply 1+log(tf) instead of raw tf
        min_df=2,               # ignore terms appearing in <2 docs
        max_df=0.95,            # ignore near-universal terms
        ngram_range=(1, 2),     # unigrams + bigrams
        max_features=100_000,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",  # mirrors transformer class-weight strategy
        solver="lbfgs",
        random_state=42,
    )),
])

lr_pipeline.fit(X_train, y_train)

# Validation set — used only for reporting, not for tuning
val_proba_lr  = lr_pipeline.predict_proba(X_val)[:, 1]
print("\n--- Validation set ---")
evaluate("LR val", y_val, val_proba_lr)

# Test set — primary comparison point
test_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]
print("\n--- Test set ---")
lr_results = evaluate("LR test", y_test, test_proba_lr)

print("\nConfusion matrix (threshold=0.35):")
print(confusion_matrix(y_test, (test_proba_lr >= 0.35).astype(int)))
print("\nClassification report (threshold=0.35):")
print(classification_report(
    y_test,
    (test_proba_lr >= 0.35).astype(int),
    target_names=["ham", "phishing"],
    zero_division=0,
))

# ── 7. Model B: TF-IDF + Multinomial Naive Bayes ────────────────────────────
print("=" * 60)
print("MODEL B: TF-IDF + Multinomial Naive Bayes")
print("=" * 60)

nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        sublinear_tf=False,     # MNB expects non-negative; skip log transform
        use_idf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=100_000,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

nb_pipeline.fit(X_train, y_train)

val_proba_nb  = nb_pipeline.predict_proba(X_val)[:, 1]
print("\n--- Validation set ---")
evaluate("NB val", y_val, val_proba_nb)

test_proba_nb = nb_pipeline.predict_proba(X_test)[:, 1]
print("\n--- Test set ---")
nb_results = evaluate("NB test", y_test, test_proba_nb)

print("\nConfusion matrix (threshold=0.35):")
print(confusion_matrix(y_test, (test_proba_nb >= 0.35).astype(int)))
print("\nClassification report (threshold=0.35):")
print(classification_report(
    y_test,
    (test_proba_nb >= 0.35).astype(int),
    target_names=["ham", "phishing"],
    zero_division=0,
))

# ── 8. Summary comparison table ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARATIVE SUMMARY (Test Set, threshold=0.35)")
print("=" * 60)
print(f"\n{'Model':<35} {'Accuracy':>10} {'F1':>8} {'Precision':>12} {'Recall':>9}")
print("-" * 75)

models_summary = {
    "TF-IDF + Logistic Regression":  lr_results.get(0.35, {}),
    "TF-IDF + Multinomial NB":        nb_results.get(0.35, {}),
}

for name, res in models_summary.items():
    if res:
        print(
            f"{name:<35} "
            f"{res['accuracy']:>10.4f} "
            f"{res['f1']:>8.4f} "
            f"{res['precision']:>12.4f} "
            f"{res['recall']:>9.4f}"
        )

print("\n  (For comparison — transformer results from thesis)")
print(f"{'ELECTRA-large (threshold=0.35)':<35} {'0.9256':>10} {'0.9055':>8} {'0.9195':>12} {'0.8920':>9}")
print(f"{'RoBERTa-large (threshold=0.35)':<35} {'0.9336':>10} {'0.9150':>8} {'0.9371':>12} {'0.8940':>9}")

# ── 9. Save results to JSON ───────────────────────────────────────────────────
output = {
    "baselines": {
        "tfidf_lr": {
            "threshold_0.5":  lr_results.get(0.5),
            "threshold_0.35": lr_results.get(0.35),
            "roc_auc":        lr_results.get("roc_auc"),
            "pr_auc":         lr_results.get("pr_auc"),
        },
        "tfidf_nb": {
            "threshold_0.5":  nb_results.get(0.5),
            "threshold_0.35": nb_results.get(0.35),
            "roc_auc":        nb_results.get("roc_auc"),
            "pr_auc":         nb_results.get("pr_auc"),
        },
    },
    "transformer_reference": {
        "electra_large": {
            "threshold_0.5":  {"accuracy": 0.9256, "f1": 0.9051, "precision": 0.9230, "recall": 0.8880},
            "threshold_0.35": {"accuracy": 0.9256, "f1": 0.9055, "precision": 0.9195, "recall": 0.8920},
        },
        "roberta_large": {
            "threshold_0.5":  {"accuracy": 0.9352, "f1": 0.9169, "precision": 0.9410, "recall": 0.8940},
            "threshold_0.35": {"accuracy": 0.9336, "f1": 0.9150, "precision": 0.9371, "recall": 0.8940},
        },
    },
}

with open("baseline_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to baseline_results.json")
print("=" * 60)