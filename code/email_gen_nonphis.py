import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
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
            print("\nDataset info:")
            print(self.data.info())
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_data(self):
        print("\nPreparing data...")
        
        if 'text' not in self.data.columns:
            raise ValueError("No 'text' column found in dataset")
        if 'label' not in self.data.columns:
            raise ValueError("No 'label' column found in dataset")
            
        # Extract text features
        self.X = self.data['text'].fillna('')
        
        # Extract phishing labels
        self.y = self.data['label']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        print(f"Target distribution (%):\n{self.y.value_counts(normalize=True) * 100}")
        
        return True
    
    def split_data(self, test_size=0.2, random_state=42):
        print(f"\nSplitting data into {int((1-test_size)*100)}/{int(test_size*100)} train/test split...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y  # Maintain class distribution in splits
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"\nTraining set distribution:\n{self.y_train.value_counts()}")
        print(f"Test set distribution:\n{self.y_test.value_counts()}")
        
        return True
    
    def vectorize_text(self, max_features=10000):
        print(f"\nVectorizing text with TF-IDF (max_features={max_features})...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)
        
        print(f"Training features shape: {self.X_train_vectorized.shape}")
        print(f"Test features shape: {self.X_test_vectorized.shape}")
        
        return True
    
    def train_models(self):
        print("\n" + "="*80)
        print("TRAINING SUPERVISED MODELS")
        print("="*80)
        
        # Define models to train
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'SVM': SVC(
                kernel='linear',
                random_state=42,
                class_weight='balanced',
                probability=True
            )
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(self.X_train_vectorized, self.y_train)
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
        
        return True
    
    def evaluate_models(self):
        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"Evaluating {name}")
            print('='*80)
            
            try:
                # Make predictions
                y_pred = model.predict(self.X_test_vectorized)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    self.y_test, y_pred, average='weighted', zero_division=0
                )
                
                # Store results
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred
                }
                
                # Print results
                print(f"\nAccuracy:  {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1 Score:  {f1:.4f}")
                
                # Detailed classification report
                print("\nDetailed Classification Report:")
                print(classification_report(self.y_test, y_pred, zero_division=0))
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                print("\nConfusion Matrix:")
                print(cm)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, self.X_train_vectorized, self.y_train, 
                    cv=5, scoring='f1_weighted'
                )
                print(f"\nCross-validation F1 scores: {cv_scores}")
                print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                self.results[name]['cv_f1_mean'] = cv_scores.mean()
                self.results[name]['cv_f1_std'] = cv_scores.std()
                
            except Exception as e:
                print(f"✗ Error evaluating {name}: {e}")
                self.results[name] = {'error': str(e)}
        
        return True
    
    def generate_summary(self):
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        # Create summary dataframe
        summary_data = []
        for name, results in self.results.items():
            if 'error' not in results:
                row = {
                    'Model': name,
                    'Accuracy': f"{results['accuracy']:.4f}",
                    'Precision': f"{results['precision']:.4f}",
                    'Recall': f"{results['recall']:.4f}",
                    'F1 Score': f"{results['f1']:.4f}",
                    'CV F1 (mean)': f"{results['cv_f1_mean']:.4f}",
                    'CV F1 (std)': f"{results['cv_f1_std']:.4f}"
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Save results
        os.makedirs('code/results', exist_ok=True)
        summary_df.to_csv('code/results/phishing_classification_results.csv', index=False)
        print(f"\n✓ Results saved to 'code/results/phishing_classification_results.csv'")
        
        # Find best model
        best_model_idx = summary_df['F1 Score'].astype(float).idxmax()
        best_model = summary_df.loc[best_model_idx]
        
        print(f"\n Best Model: {best_model['Model']}")
        print(f"   Accuracy:  {best_model['Accuracy']}")
        print(f"   Precision: {best_model['Precision']}")
        print(f"   Recall:    {best_model['Recall']}")
        print(f"   F1 Score:  {best_model['F1 Score']}")
        
        return True
    
    def run_pipeline(self):
        print("="*80)
        print("STARTING MACHINE LEARNING PIPELINE")
        print("="*80)
        
        steps = [
            self.load_data,
            self.prepare_data,
            self.split_data,
            self.vectorize_text,
            self.train_models,
            self.evaluate_models,
            self.generate_summary
        ]
        
        for step in steps:
            if not step():
                print(f"\n ERROR: Pipeline failed at step: {step.__name__}")
                return False
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        return True

def main():
    dataset_path = 'code/combined_dataset.csv'
    
    pipeline = MLPipeline(dataset_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()