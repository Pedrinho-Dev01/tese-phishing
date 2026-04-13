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
    confusion_matrix, 
    precision_recall_fscore_support, 
    accuracy_score) 

from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
import pandas as pd
import numpy as np
import torch

torch.cuda.empty_cache()

# 1. Load dataset
df = pd.read_csv('code/combined_emails_dataset.csv')

# 2. Prepare for transformers
df['labels'] = df['label'].map({'non-phishing': 0, 'phishing': 1})

print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print(df['labels'].value_counts())
print(f"Class 0: {(df['labels']==0).sum()} ({(df['labels']==0).mean()*100:.1f}%)")
print(f"Class 1: {(df['labels']==1).sum()} ({(df['labels']==1).mean()*100:.1f}%)")
print("="*60 + "\n")

# 3. Split train/val/test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['labels'], random_state=42)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# 4. Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['labels']),
    y=train_df['labels']
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"\nClass weights: {class_weights}")

# 5. Load RoBERTa-large
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=2
)

# gradient_checkpointing disabled â€” A6000 has enough VRAM, runs faster without it

data_collator = DataCollatorWithPadding(tokenizer)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
val_dataset = Dataset.from_pandas(val_df[['text', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# 6. Training arguments
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir='./results_roberta_large',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,             # Slightly lower than base-optimised 4e-5; large models prefer more conservative LR
    per_device_train_batch_size=64, # Reduced from 128: large model activations are bigger
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    greater_is_better=True,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    label_smoothing_factor=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='./logs_roberta_large',
    logging_steps=20,
    logging_first_step=True,
    bf16=use_bf16,
    fp16=not use_bf16 and torch.cuda.is_available(),
    use_cpu=False,
    gradient_checkpointing=False,   # Disable: enough VRAM, faster without it
    optim="adamw_torch_fused",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,       # Parallel data loading
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
print("Starting RoBERTa-large training")
print("="*60)
trainer.train()

# 9. Test set evaluation
print("\n" + "="*60)
print("Evaluating on test set...")
print("="*60)

preds = trainer.predict(test_dataset)
y_true = preds.label_ids
y_pred_standard = preds.predictions.argmax(-1)

print("\n--- RESULTS WITH STANDARD THRESHOLD (0.5) ---")
print(confusion_matrix(y_true, y_pred_standard))
print(classification_report(y_true, y_pred_standard, zero_division=0))

precision_std, recall_std, f1_std, _ = precision_recall_fscore_support(
    y_true, y_pred_standard, average='binary', zero_division=0
)
acc_std = accuracy_score(y_true, y_pred_standard)

print(f"\n{'='*60}")
print("FINAL TEST RESULTS - RoBERTa-Large")
print(f"{'='*60}")
print("\nStandard Threshold (0.5):")
print(f"  Accuracy:  {acc_std:.4f}")
print(f"  F1 Score:  {f1_std:.4f}")
print(f"  Precision: {precision_std:.4f}")
print(f"  Recall:    {recall_std:.4f}")
print(f"{'='*60}")

# 10. Save model
print("\nSaving model...")
trainer.save_model('./ml_code/models/roberta_large_final')
tokenizer.save_pretrained('./ml_code/models/roberta_large_final')

import json
threshold_info = {
    'evaluation_threshold': 0.5,
    'metrics': {'accuracy': acc_std, 'f1': f1_std, 'precision': precision_std, 'recall': recall_std}
}
with open('./ml_code/models/roberta_large_final/threshold_config.json', 'w') as f:
    json.dump(threshold_info, f, indent=2)

print("Model saved to './ml_code/models/roberta_large_final'")