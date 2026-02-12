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


from datasets import Dataset
import pandas as pd
import numpy as np
from scipy.special import softmax

# 1. Load dataset
df = pd.read_csv('code/combined_emails_dataset.csv')

# 2. Prepare for transformers
df['labels'] = df['label'].map({'non-phishing': 0, 'phishing': 1})

# 3. Split train/val/test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['labels'], random_state=42)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# 4. Load Stuff
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=2
)
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
    probs = softmax(pred.predictions, axis=1)[:, 1]

    # Tune threshold (example: 0.35)
    preds = (probs >= 0.35).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 5. Training arguments
training_args = TrainingArguments(
    output_dir='./results_roberta',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    greater_is_better=True,
    num_train_epochs=8,
    weight_decay=0.01,
    warmup_steps=100, 
    lr_scheduler_type="cosine",
    label_smoothing_factor=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='./logs_roberta',
    logging_steps=10,
    fp16=True,              # Enable mixed precision training
    use_cpu=False,          # Use GPU
)

# 6. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    data_collator=data_collator
)

# 7. Train and evaluate
print("\nStarting training...")
trainer.train()

# 8. Test set evaluation
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset)

preds = trainer.predict(test_dataset)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(-1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

print(f"\n{'='*60}")
print("FINAL TEST RESULTS")
print(f"{'='*60}")
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall:    {test_results['eval_recall']:.4f}")
print(f"{'='*60}")

# 9. Save model
print("\nSaving model...")
trainer.save_model('./ml_code/models/roberta_final')
tokenizer.save_pretrained('./ml_code/models/roberta_final')
print("✓ Model saved to './ml_code/models/roberta_final'")