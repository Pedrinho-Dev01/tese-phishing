from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)
from datasets import Dataset
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid
import torch
import json

torch.cuda.empty_cache()

# 1. Load dataset — keep only phishing emails
df = pd.read_csv('code/combined_emails_dataset.csv')
df = df[df['label'] == 'phishing'].copy()
print(f"\nPhishing-only subset: {len(df):,} emails")

# 2. Parse multi-label emotions (e.g. "curiosity#excitement#fear")
df = df.dropna(subset=['emotion'])
df['emotion_list'] = df['emotion'].str.split('#')

emotion_classes = sorted(set(e for emotions in df['emotion_list'] for e in emotions))
label2id = {e: i for i, e in enumerate(emotion_classes)}
id2label  = {i: e for e, i in label2id.items()}
num_labels = len(emotion_classes)

def encode_labels(emotion_list):
    vec = np.zeros(num_labels, dtype=np.float32)
    for e in emotion_list:
        if e in label2id:
            vec[label2id[e]] = 1.0
    return vec

df['labels'] = df['emotion_list'].apply(encode_labels)

print("\n" + "="*60)
print("EMOTION CLASS DISTRIBUTION (multi-label)")
print("="*60)
label_matrix = np.stack(df['labels'].values)
for i, emotion in enumerate(emotion_classes):
    count = int(label_matrix[:, i].sum())
    print(f"  {emotion:<30} {count:>6,}  ({count/len(df)*100:.1f}%)")
print(f"\nTotal classes: {num_labels}")
avg_labels = label_matrix.sum(axis=1).mean()
print(f"Avg labels per sample: {avg_labels:.2f}")
print("="*60 + "\n")

# 3. Split train/val/test
from sklearn.model_selection import train_test_split
primary_label = label_matrix.argmax(axis=1)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=primary_label, random_state=42)
train_primary = np.stack(train_df['labels'].values).argmax(axis=1)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_primary, random_state=42)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# 4. Compute pos_weight with sqrt dampening + max clamp
train_label_matrix = np.stack(train_df['labels'].values)
pos_counts = train_label_matrix.sum(axis=0)
neg_counts = len(train_df) - pos_counts

raw_weights = neg_counts / np.clip(pos_counts, 1, None)
dampened_weights = np.sqrt(raw_weights)
pos_weight = torch.tensor(dampened_weights, dtype=torch.float32).clamp(max=5.0)

print(f"\nRaw pos_weight:      {raw_weights.round(2)}")
print(f"Dampened pos_weight: {pos_weight.numpy().round(2)}")

# 5. Load ELECTRA-large
tokenizer = AutoTokenizer.from_pretrained("google/electra-large-discriminator")
model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-large-discriminator",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification"
)

model.gradient_checkpointing_enable()
data_collator = DataCollatorWithPadding(tokenizer)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

def to_dataset(dataframe):
    records = {
        'text':   dataframe['text'].tolist(),
        'labels': [row.tolist() for row in dataframe['labels'].values]
    }
    ds = Dataset.from_dict(records)
    return ds.map(tokenize_function, batched=True)

train_dataset = to_dataset(train_df)
val_dataset   = to_dataset(val_df)
test_dataset  = to_dataset(test_df)

THRESHOLD = 0.5

def compute_metrics(pred):
    probs = sigmoid(pred.predictions)
    # No fallback — raw threshold only
    preds = (probs >= THRESHOLD).astype(int)
    labels = pred.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    exact_match = accuracy_score(labels, preds)

    return {
        'exact_match': exact_match,
        'f1_macro':    f1,
        'precision':   precision,
        'recall':      recall
    }

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

# 6. Training arguments
training_args = TrainingArguments(
    output_dir='./results_electra_emotion',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    greater_is_better=True,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_ratio=0.15,
    lr_scheduler_type="cosine",
    label_smoothing_factor=0.0,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir='./logs_electra_emotion',
    logging_steps=50,
    logging_first_step=True,
    bf16=True,
    use_cpu=False,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    dataloader_pin_memory=False,
    save_total_limit=2,
    report_to="none",
)

# 7. Create trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator=data_collator
)

# 8. Train
print("\n" + "="*60)
print("Starting ELECTRA-large multi-label emotion training")
print("="*60)
trainer.train()

# ── 9. Tune threshold on validation set ──────────────────────────────────────
print("\n" + "="*60)
print("Tuning classification threshold on validation set (NO fallback)...")
print("="*60)

val_output   = trainer.predict(val_dataset)
val_probs    = sigmoid(val_output.predictions)
val_true     = val_output.label_ids

# --- Global threshold search (no fallback) ---
best_global_t, best_global_f1 = 0.5, 0.0
print(f"\n{'Threshold':<12} {'F1 Macro':<12} {'Precision':<12} {'Recall':<12} {'Exact Match':<12} {'Empty Preds'}")
print("-" * 72)

for t in np.arange(0.10, 0.81, 0.05):
    preds = (val_probs >= t).astype(int)
    empty = (preds.sum(axis=1) == 0).sum()
    p, r, f1, _ = precision_recall_fscore_support(val_true, preds, average='macro', zero_division=0)
    em = accuracy_score(val_true, preds)
    print(f"  {t:.2f}        {f1:.4f}       {p:.4f}       {r:.4f}       {em:.4f}       {empty}")
    if f1 > best_global_f1:
        best_global_f1 = f1
        best_global_t  = t

print(f"\n✓ Best global threshold: {best_global_t:.2f}  →  Val F1 Macro: {best_global_f1:.4f}")

# --- Per-class threshold search (no fallback) ---
print("\n" + "="*60)
print("Per-class threshold search...")
print("="*60)

per_class_thresholds = []
for cls_i in range(num_labels):
    best_t_cls, best_f1_cls = 0.5, 0.0
    for t in np.arange(0.10, 0.81, 0.05):
        preds_cls = (val_probs[:, cls_i] >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            val_true[:, cls_i], preds_cls, average='binary', zero_division=0
        )
        if f1 > best_f1_cls:
            best_f1_cls = f1
            best_t_cls  = t
    per_class_thresholds.append(best_t_cls)
    print(f"  {emotion_classes[cls_i]:<30} best_t={best_t_cls:.2f}  F1={best_f1_cls:.4f}")

per_class_thresholds = np.array(per_class_thresholds)

# ── 10. Test set evaluation ───────────────────────────────────────────────────
print("\n" + "="*60)
print("Evaluating on test set (NO fallback)...")
print("="*60)

preds_output = trainer.predict(test_dataset)
y_true = preds_output.label_ids
probs  = sigmoid(preds_output.predictions)

y_pred_global   = (probs >= best_global_t).astype(int)
y_pred_perclass = (probs >= per_class_thresholds).astype(int)

# Report empty prediction counts honestly
empty_global   = (y_pred_global.sum(axis=1) == 0).sum()
empty_perclass = (y_pred_perclass.sum(axis=1) == 0).sum()
print(f"\nSamples with NO predicted label — global: {empty_global}, per-class: {empty_perclass}")

def report(y_true, y_pred, label):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    em = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS — {label}")
    print(f"{'='*60}")
    print(f"  Exact Match Accuracy: {em:.4f}")
    print(f"  Macro F1:             {f1:.4f}")
    print(f"  Macro Precision:      {p:.4f}")
    print(f"  Macro Recall:         {r:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=emotion_classes, zero_division=0))
    return {'exact_match': em, 'f1_macro': f1, 'precision_macro': p, 'recall_macro': r}

metrics_global   = report(y_true, y_pred_global,   f"Global threshold ({best_global_t:.2f})")
metrics_perclass = report(y_true, y_pred_perclass, "Per-class thresholds")

# ── 11. Misclassified examples ────────────────────────────────────────────────
print("\n" + "="*60)
print("MISCLASSIFIED EXAMPLES — per-class threshold (up to 5)")
print("="*60)

test_texts = test_df['text'].values
wrong_idx  = np.where((y_true != y_pred_perclass).any(axis=1))[0]
print(f"\nTotal misclassified: {len(wrong_idx)} / {len(y_true)}")

for i, idx in enumerate(wrong_idx[:5]):
    true_emotions = [emotion_classes[j] for j in range(num_labels) if y_true[idx][j] == 1]
    pred_emotions = [emotion_classes[j] for j in range(num_labels) if y_pred_perclass[idx][j] == 1]
    print(f"\n[Example {i+1}]")
    print(f"  True: {', '.join(true_emotions)}")
    print(f"  Pred: {', '.join(pred_emotions) or 'none'}")
    print(f"  Text: {test_texts[idx][:400]}...")
    print("-"*60)

# ── 12. Save model ────────────────────────────────────────────────────────────
print("\nSaving model...")
trainer.save_model('./ml_code_emotion/models/electra_emotion_final')
tokenizer.save_pretrained('./ml_code_emotion/models/electra_emotion_final')

model_info = {
    'num_labels':   num_labels,
    'label2id':     label2id,
    'id2label':     id2label,
    'threshold_global':    float(best_global_t),
    'threshold_per_class': {emotion_classes[i]: float(per_class_thresholds[i])
                            for i in range(num_labels)},
    'problem_type': 'multi_label_classification',
    'pos_weight_strategy': 'sqrt_dampened_clamp5',
    'apply_threshold_fallback': False,
    'test_metrics_global': metrics_global,
    'test_metrics_perclass': metrics_perclass,
}
with open('./ml_code_emotion/models/electra_emotion_final/model_config.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Model saved to './ml_code_emotion/models/electra_emotion_final'")
print("\nPer-class thresholds saved to model_config.json")
print("Use 'threshold_per_class' at inference time for best results.")