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
from scipy.special import expit as sigmoid  # sigmoid instead of softmax for multi-label
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

# 3. Split train/val/test (stratify on most common label per sample as proxy)
from sklearn.model_selection import train_test_split
primary_label = label_matrix.argmax(axis=1)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=primary_label, random_state=42)
train_primary = np.stack(train_df['labels'].values).argmax(axis=1)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_primary, random_state=42)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# 4. Compute per-class pos_weight for BCEWithLogitsLoss
train_label_matrix = np.stack(train_df['labels'].values)
pos_counts = train_label_matrix.sum(axis=0)
neg_counts = len(train_df) - pos_counts
pos_weight = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float32)
print(f"\nPos weights: {pos_weight}")

# 5. Load ELECTRA-base
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-base-discriminator",
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

        # BCEWithLogitsLoss for multi-label (vs CrossEntropyLoss for single-label)
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
    label_smoothing_factor=0.0,     # must be 0 — incompatible with BCEWithLogitsLoss
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
print("Starting ELECTRA-base multi-label emotion training")
print("="*60)
trainer.train()

# 9. Test set evaluation
print("\n" + "="*60)
print("Evaluating on test set...")
print("="*60)

preds_output = trainer.predict(test_dataset)
y_true = preds_output.label_ids
probs  = sigmoid(preds_output.predictions)
y_pred = (probs >= THRESHOLD).astype(int)

print("\n--- PER-CLASS REPORT ---")
print(classification_report(
    y_true, y_pred,
    target_names=emotion_classes,
    zero_division=0
))

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro', zero_division=0
)
exact_match = accuracy_score(y_true, y_pred)

print(f"\n{'='*60}")
print("FINAL TEST RESULTS - ELECTRA-base Multi-Label Emotion")
print(f"{'='*60}")
print(f"  Exact Match Accuracy: {exact_match:.4f}")
print(f"  Macro F1:             {f1:.4f}")
print(f"  Macro Precision:      {precision:.4f}")
print(f"  Macro Recall:         {recall:.4f}")
print(f"{'='*60}")

# 10. Misclassified examples
print("\n" + "="*60)
print("MISCLASSIFIED EXAMPLES (up to 5)")
print("="*60)

test_texts = test_df['text'].values
wrong_idx  = np.where((y_true != y_pred).any(axis=1))[0]
print(f"\nTotal misclassified: {len(wrong_idx)} / {len(y_true)}")

for i, idx in enumerate(wrong_idx[:5]):
    true_emotions = [emotion_classes[j] for j in range(num_labels) if y_true[idx][j] == 1]
    pred_emotions = [emotion_classes[j] for j in range(num_labels) if y_pred[idx][j] == 1]
    print(f"\n[Example {i+1}]")
    print(f"  True: {', '.join(true_emotions)}")
    print(f"  Pred: {', '.join(pred_emotions) or 'none'}")
    print(f"  Text: {test_texts[idx][:400]}...")
    print("-"*60)

# 11. Save model
print("\nSaving model...")
trainer.save_model('./ml_code/models/electra_emotion_final')
tokenizer.save_pretrained('./ml_code/models/electra_emotion_final')

model_info = {
    'num_labels':   num_labels,
    'label2id':     label2id,
    'id2label':     id2label,
    'threshold':    THRESHOLD,
    'problem_type': 'multi_label_classification',
    'test_metrics': {
        'exact_match_accuracy': exact_match,
        'f1_macro':             f1,
        'precision_macro':      precision,
        'recall_macro':         recall
    }
}
with open('./ml_code/models/electra_emotion_final/model_config.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Model saved to './ml_code/models/electra_emotion_final'")