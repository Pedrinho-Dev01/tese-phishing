#!/usr/bin/env python3
"""
Supervised Machine Learning Pipeline for Email Classification
Trains models for phishing detection and emotion classification
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SupervisedMLPipeline:
    def __init__(self, data_path):
        """Initialize the supervised ML pipeline"""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y_phishing = None
        self.y_emotion = None
        
        # Split data
        self.X_train = None
        self.X_test = None
        self.y_phishing_train = None
        self.y_phishing_test = None
        self.y_emotion_train = None
        self.y_emotion_test = None
        
        # Models
        self.phishing_models = {}
        self.emotion_models = {}
        self.phishing_results = {}
        self.emotion_results = {}
        
    def load_data(self):
        """Load the combined dataset"""
        print("Loading dataset...")
        try:
            possible_files = [
                self.data_path,
                'code/combined_dataset.csv',
                'code/combined_emails_dataset.csv',
            ]
            
            for file_path in possible_files:
                try:
                    self.data = pd.read_csv(file_path)
                    print(f"✓ Successfully loaded data from: {file_path}")
                    break
                except FileNotFoundError:
                    continue
            
            if self.data is None:
                raise FileNotFoundError("Could not find dataset file")
                
            print(f"Dataset shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def normalize_text(self, text):
        """Normalize text by standardizing whitespace and formatting"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        text = re.sub(r'[""''`]', '"', text)
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        return text
    
    def prepare_data(self):
        """Prepare features and labels"""
        print("\nPreparing data...")
        
        if 'text' not in self.data.columns:
            raise ValueError("No 'text' column found in dataset")
        
        # Normalize text
        print("Normalizing text...")
        self.X = self.data['text'].apply(self.normalize_text)
        
        # Extract labels
        if 'label' not in self.data.columns:
            raise ValueError("No 'label' column found for phishing detection")
        if 'emotion' not in self.data.columns:
            raise ValueError("No 'emotion' column found for emotion classification")
        
        self.y_phishing = self.data['label']
        self.y_emotion = self.data['emotion']
        
        # Show label distributions
        print(f"\nPhishing label distribution:")
        print(self.y_phishing.value_counts())
        
        print(f"\nEmotion label distribution:")
        print(self.y_emotion.value_counts())
        
        print(f"\nTotal samples: {len(self.X)}")
        
        return True
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data (80/20 train/test)...")
        
        # Split for phishing detection
        self.X_train, self.X_test, self.y_phishing_train, self.y_phishing_test = train_test_split(
            self.X, self.y_phishing, test_size=test_size, random_state=random_state, stratify=self.y_phishing
        )
        
        # For emotion classification, we'll use the same split
        # but we need to align the indices
        train_idx = self.X_train.index
        test_idx = self.X_test.index
        
        self.y_emotion_train = self.y_emotion.loc[train_idx]
        self.y_emotion_test = self.y_emotion.loc[test_idx]
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        print(f"\nPhishing training distribution:")
        print(self.y_phishing_train.value_counts())
        
        print(f"\nEmotion training distribution:")
        print(self.y_emotion_train.value_counts())
        
        return True
    
    def train_phishing_models(self):
        """Train models for phishing detection"""
        print("\n" + "="*70)
        print("TRAINING PHISHING DETECTION MODELS")
        print("="*70)
        
        # Define models with pipelines
        self.phishing_models = {
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            
            'Naive Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('clf', MultinomialNB())
            ]),
            
            'Random Forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ]),
            
            'SVM': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
                ('clf', SVC(kernel='linear', random_state=42))
            ]),
            
            'Gradient Boosting': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
                ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ])
        }
        
        # Train each model
        for name, model in self.phishing_models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(self.X_train, self.y_phishing_train)
                
                # Quick cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_phishing_train, cv=5, scoring='f1_weighted')
                print(f"  ✓ Trained successfully")
                print(f"  Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")
        
        return True
    
    def train_emotion_models(self):
        """Train models for emotion classification (phishing emails only)"""
        print("\n" + "="*70)
        print("TRAINING EMOTION CLASSIFICATION MODELS (PHISHING EMAILS ONLY)")
        print("="*70)
        
        # Filter to only phishing emails
        phishing_mask_train = self.y_phishing_train == 'phishing'
        phishing_mask_test = self.y_phishing_test == 'phishing'
        
        X_train_phishing = self.X_train[phishing_mask_train]
        X_test_phishing = self.X_test[phishing_mask_test]
        y_emotion_train_phishing = self.y_emotion_train[phishing_mask_train]
        y_emotion_test_phishing = self.y_emotion_test[phishing_mask_test]
        
        print(f"Training on {len(X_train_phishing)} phishing emails")
        print(f"Testing on {len(X_test_phishing)} phishing emails")
        
        # Store filtered data for later evaluation
        self.X_test_phishing = X_test_phishing
        self.y_emotion_test_phishing = y_emotion_test_phishing
        
        # Define models
        self.emotion_models = {
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')),
                ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
            ]),
            
            'Naive Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('clf', MultinomialNB(alpha=0.1))
            ]),
            
            'Random Forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
                ('clf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
            ]),
            
            'SVM': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
                ('clf', SVC(kernel='rbf', random_state=42, C=1.0))
            ]),
            
            'Gradient Boosting': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
                ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ])
        }
        
        # Train each model
        for name, model in self.emotion_models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(X_train_phishing, y_emotion_train_phishing)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_phishing, y_emotion_train_phishing, 
                                          cv=5, scoring='f1_weighted')
                print(f"  ✓ Trained successfully")
                print(f"  Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")
        
        return True
    
    def evaluate_phishing_models(self):
        """Evaluate phishing detection models"""
        print("\n" + "="*70)
        print("EVALUATING PHISHING DETECTION MODELS")
        print("="*70)
        
        for name, model in self.phishing_models.items():
            print(f"\n{name}:")
            print("-" * 50)
            
            try:
                # Predictions
                y_pred = model.predict(self.X_test)
                
                # Metrics
                accuracy = accuracy_score(self.y_phishing_test, y_pred)
                f1 = f1_score(self.y_phishing_test, y_pred, average='weighted')
                precision, recall, _, _ = precision_recall_fscore_support(
                    self.y_phishing_test, y_pred, average='weighted', zero_division=0
                )
                
                print(f"Accuracy:  {accuracy:.4f}")
                print(f"F1 Score:  {f1:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                
                # Store results
                self.phishing_results[name] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred
                }
                
                # Classification report
                print("\nClassification Report:")
                print(classification_report(self.y_phishing_test, y_pred, zero_division=0))
                
                # Confusion matrix
                cm = confusion_matrix(self.y_phishing_test, y_pred)
                print("Confusion Matrix:")
                print(cm)
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return True
    
    def evaluate_emotion_models(self):
        """Evaluate emotion classification models"""
        print("\n" + "="*70)
        print("EVALUATING EMOTION CLASSIFICATION MODELS")
        print("="*70)
        
        for name, model in self.emotion_models.items():
            print(f"\n{name}:")
            print("-" * 50)
            
            try:
                # Predictions
                y_pred = model.predict(self.X_test_phishing)
                
                # Metrics
                accuracy = accuracy_score(self.y_emotion_test_phishing, y_pred)
                f1 = f1_score(self.y_emotion_test_phishing, y_pred, average='weighted')
                precision, recall, _, _ = precision_recall_fscore_support(
                    self.y_emotion_test_phishing, y_pred, average='weighted', zero_division=0
                )
                
                print(f"Accuracy:  {accuracy:.4f}")
                print(f"F1 Score:  {f1:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                
                # Store results
                self.emotion_results[name] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred
                }
                
                # Classification report
                print("\nClassification Report:")
                print(classification_report(self.y_emotion_test_phishing, y_pred, zero_division=0))
                
                # Confusion matrix
                cm = confusion_matrix(self.y_emotion_test_phishing, y_pred)
                print("Confusion Matrix:")
                print(cm)
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return True
    
    def generate_summary(self):
        """Generate summary report"""
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        # Create results directory
        os.makedirs('code/results', exist_ok=True)
        
        # Phishing detection summary
        print("\n" + "="*70)
        print("PHISHING DETECTION RESULTS")
        print("="*70)
        
        phishing_data = []
        for name, results in self.phishing_results.items():
            phishing_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1': f"{results['f1']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}"
            })
        
        phishing_df = pd.DataFrame(phishing_data)
        print(phishing_df.to_string(index=False))
        
        # Find best model
        best_phishing = max(self.phishing_results.items(), key=lambda x: x[1]['f1'])
        print(f"\nBEST MODEL: {best_phishing[0]}")
        print(f"F1 Score: {best_phishing[1]['f1']:.4f}")
        
        phishing_df.to_csv('code/results/supervised_phishing_results.csv', index=False)
        
        # Emotion classification summary
        print("\n" + "="*70)
        print("EMOTION CLASSIFICATION RESULTS (PHISHING EMAILS ONLY)")
        print("="*70)
        
        emotion_data = []
        for name, results in self.emotion_results.items():
            emotion_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1': f"{results['f1']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}"
            })
        
        emotion_df = pd.DataFrame(emotion_data)
        print(emotion_df.to_string(index=False))
        
        # Find best model
        best_emotion = max(self.emotion_results.items(), key=lambda x: x[1]['f1'])
        print(f"\nBEST MODEL: {best_emotion[0]}")
        print(f"F1 Score: {best_emotion[1]['f1']:.4f}")
        
        emotion_df.to_csv('code/results/supervised_emotion_results.csv', index=False)
        
        print("\n" + "="*70)
        print("Results saved to 'code/results/' directory")
        print("="*70)
        
        return True
    
    def run_pipeline(self):
        """Run the complete supervised ML pipeline"""
        print("="*80)
        print("SUPERVISED MACHINE LEARNING PIPELINE")
        print("="*80)
        
        steps = [
            ('Loading data', self.load_data),
            ('Preparing data', self.prepare_data),
            ('Splitting data', self.split_data),
            ('Training phishing models', self.train_phishing_models),
            ('Training emotion models', self.train_emotion_models),
            ('Evaluating phishing models', self.evaluate_phishing_models),
            ('Evaluating emotion models', self.evaluate_emotion_models),
            ('Generating summary', self.generate_summary)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*80}")
            print(f"STEP: {step_name.upper()}")
            print('='*80)
            if not step_func():
                print(f"\nPipeline failed at: {step_name}")
                return False
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        return True

def main():
    """Main function"""
    dataset_path = 'code/combined_emails_dataset.csv'
    
    pipeline = SupervisedMLPipeline(dataset_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()