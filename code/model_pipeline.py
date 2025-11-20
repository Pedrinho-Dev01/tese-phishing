#!/usr/bin/env python3
"""
Machine Learning Pipeline for Combined Dataset
Splits data, trains models, and evaluates performance
"""

import pandas as pd
import numpy as np
import os
import re
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
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self, data_path):
        """Initialize the ML pipeline with data path"""
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
        """Load the combined dataset"""
        print("Loading dataset...")
        try:
            # Try different possible file formats/names
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
            
            # Show data info
            print("\nDataset info:")
            print(self.data.info())
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def normalize_text(self, text):
        """Normalize text by standardizing whitespace and formatting"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Replace multiple whitespaces (including \n, \r, \t) with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize common formatting patterns
        # Remove extra punctuation spacing
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        
        # Normalize quotation marks
        text = re.sub(r'[""''`]', '"', text)
        
        # Remove excessive punctuation (more than 2 consecutive)
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        return text
    
    def prepare_data(self):
        """Prepare features and store labels for validation only"""
        print("\nPreparing data...")
        
        if 'text' not in self.data.columns:
            raise ValueError("No 'text' column found in dataset")
            
        # Extract and normalize text features
        print("Normalizing text to remove formatting differences...")
        self.X = self.data['text'].apply(self.normalize_text)
        print(f"Features shape: {self.X.shape}")
        print("✓ Text normalization completed")
        
        # Store labels for validation (but don't use them for training)
        self.emotion_labels = None
        self.phishing_labels = None
        
        if 'emotion' in self.data.columns:
            self.emotion_labels = self.data['emotion']
            print(f"Found emotion labels: {self.emotion_labels.value_counts().to_dict()}")
            
        # Use 'label' column for phishing labels
        if 'label' in self.data.columns:
            self.phishing_labels = self.data['label']
            print(f"Found phishing labels: {self.phishing_labels.value_counts().to_dict()}")
                
        if self.emotion_labels is None and self.phishing_labels is None:
            print("Warning: No emotion or phishing labels found for validation")
            return False
            
        print("\nNote: Labels will be used only for validation, not for training")
        return True
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets (80/20 split)"""
        print(f"\nSplitting data into {int((1-test_size)*100)}/{int(test_size*100)} train/test split...")
        
        # Split text data
        indices = np.arange(len(self.X))
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
        
        self.X_train = self.X.iloc[train_idx]
        self.X_test = self.X.iloc[test_idx]
        
        # Split labels for validation (but don't use for training)
        if self.emotion_labels is not None:
            self.emotion_train = self.emotion_labels.iloc[train_idx]
            self.emotion_test = self.emotion_labels.iloc[test_idx]
            
        if self.phishing_labels is not None:
            self.phishing_train = self.phishing_labels.iloc[train_idx]
            self.phishing_test = self.phishing_labels.iloc[test_idx]
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return True
    
    def vectorize_text(self, max_features=10000):
        """Convert text to TF-IDF features"""
        print(f"\nVectorizing normalized text with TF-IDF (max_features={max_features})...")
        print("Text preprocessing: whitespace normalization, lowercase, alphabetic tokens only")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            lowercase=True,  # Convert to lowercase
            token_pattern=r'\b[A-Za-z]{2,}\b',  # Only alphabetic tokens, min 2 chars
            strip_accents='ascii'  # Remove accents
        )
        
        # Fit on training data and transform both train and test
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)
        
        print(f"Training features shape: {self.X_train_vectorized.shape}")
        print(f"Test features shape: {self.X_test_vectorized.shape}")
        
        return True
    
    def train_models(self):
        """Train unsupervised/clustering models and supervised models for comparison"""
        print("\nTraining models...")
        print("Training models WITHOUT access to labels (unsupervised approach)")
        
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Unsupervised models that learn patterns from text only
        self.models = {
            'KMeans Clustering': KMeans(n_clusters=8, random_state=42, n_init=10),  # Assume ~8 emotion categories
            'Gaussian Mixture': GaussianMixture(n_components=8, random_state=42)
        }
        
        # Train unsupervised models
        for name, model in self.models.items():
            print(f"Training {name} (unsupervised)...")
            try:
                if 'LDA' in name:
                    # LDA needs non-negative features, use raw counts
                    from sklearn.feature_extraction.text import CountVectorizer
                    count_vec = CountVectorizer(max_features=1000, stop_words='english')
                    X_counts = count_vec.fit_transform(self.X_train)
                    model.fit(X_counts)
                    self.lda_vectorizer = count_vec  # Store for later use
                elif 'Gaussian' in name:
                    # Gaussian Mixture needs dense arrays
                    X_dense = self.X_train_vectorized.toarray()
                    model.fit(X_dense)
                else:
                    model.fit(self.X_train_vectorized)
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
        
        return True
    
    def evaluate_models(self):
        """Evaluate unsupervised models against true labels"""
        print("\nEvaluating models against TRUE LABELS...")
        print("Now we reveal the labels to see how well unsupervised models captured patterns")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {name}")
            print('='*60)
            
            try:
                # Get cluster assignments/predictions
                if 'LDA' in name:
                    X_test_counts = self.lda_vectorizer.transform(self.X_test)
                    cluster_pred = model.transform(X_test_counts).argmax(axis=1)
                elif 'Gaussian' in name:
                    # Gaussian Mixture needs dense arrays for prediction
                    X_test_dense = self.X_test_vectorized.toarray()
                    cluster_pred = model.predict(X_test_dense)
                else:
                    cluster_pred = model.predict(self.X_test_vectorized)
                
                self.results[name] = {'cluster_predictions': cluster_pred}
                
                # Only evaluate if model was trained successfully
                if name in self.models and hasattr(self.models[name], '_check_is_fitted'):
                    try:
                        # Check if model is fitted
                        self.models[name]._check_is_fitted()
                        model_is_fitted = True
                    except:
                        model_is_fitted = False
                else:
                    model_is_fitted = True  # Assume fitted for other models
                
                if not model_is_fitted:
                    print(f"Skipping evaluation - {name} was not successfully trained")
                    continue
                
                # Evaluate against emotion labels if available
                if self.emotion_labels is not None:
                    print(f"\nEMOTION CLASSIFICATION EVALUATION:")
                    emotion_metrics = self._evaluate_clustering_against_labels(
                        cluster_pred, self.emotion_test, 'emotion'
                    )
                    self.results[name]['emotion_metrics'] = emotion_metrics
                
                # Evaluate against phishing labels if available  
                if self.phishing_labels is not None:
                    print(f"\nPHISHING DETECTION EVALUATION:")
                    phishing_metrics = self._evaluate_clustering_against_labels(
                        cluster_pred, self.phishing_test, 'phishing'
                    )
                    self.results[name]['phishing_metrics'] = phishing_metrics
                    
            except Exception as e:
                print(f"✗ Error evaluating {name}: {e}")
                self.results[name] = {'error': str(e)}
        
        return True
    
    def _evaluate_clustering_against_labels(self, cluster_pred, true_labels, label_type):
        """Evaluate clustering results against true labels using basic metrics"""
        from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, accuracy_score
        from sklearn.preprocessing import LabelEncoder
        
        # Convert string labels to numeric if needed
        le = LabelEncoder()
        true_labels_encoded = le.fit_transform(true_labels)
        
        # Calculate basic metrics
        ari = adjusted_rand_score(true_labels_encoded, cluster_pred)
        
        # For F1, precision, accuracy we need to handle multi-class properly
        f1 = f1_score(true_labels_encoded, cluster_pred, average='weighted')
        precision = precision_score(true_labels_encoded, cluster_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_labels_encoded, cluster_pred)
        
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Show cluster-to-label mapping (simplified)
        print(f"\n  Cluster to {label_type} mapping:")
        mapping_df = pd.crosstab(cluster_pred, true_labels, margins=True)
        print(mapping_df)
        
        # Also show encoded mapping for clarity
        print(f"\n  Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        return {
            'ari': ari,
            'f1': f1,
            'precision': precision,
            'accuracy': accuracy,
            'mapping': mapping_df
        }
    

    
    def generate_summary(self):
        """Generate a summary report of all results"""
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY - UNSUPERVISED LEARNING EVALUATION")
        print("="*80)
        
        # Create summary dataframe for emotion classification
        if any('emotion_metrics' in results for results in self.results.values()):
            print("\nEMOTION CLASSIFICATION RESULTS:")
            emotion_data = []
            for name, results in self.results.items():
                if 'emotion_metrics' in results:
                    metrics = results['emotion_metrics']
                    row = {
                        'Model': name,
                        'ARI': f"{metrics['ari']:.4f}",
                        'F1': f"{metrics['f1']:.4f}", 
                        'Precision': f"{metrics['precision']:.4f}",
                        'Accuracy': f"{metrics['accuracy']:.4f}"
                    }
                    emotion_data.append(row)
            
            emotion_df = pd.DataFrame(emotion_data)
            print(emotion_df.to_string(index=False))
            
            # Create results directory if it doesn't exist
            import os
            os.makedirs('code/results', exist_ok=True)
            
            emotion_df.to_csv('code/results/emotion_clustering_results.csv', index=False)
        
        # Create summary dataframe for phishing detection
        if any('phishing_metrics' in results for results in self.results.values()):
            print("\nPHISHING DETECTION RESULTS:")
            phishing_data = []
            for name, results in self.results.items():
                if 'phishing_metrics' in results:
                    metrics = results['phishing_metrics']
                    row = {
                        'Model': name,
                        'ARI': f"{metrics['ari']:.4f}",
                        'F1': f"{metrics['f1']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Accuracy': f"{metrics['accuracy']:.4f}"
                    }
                    phishing_data.append(row)
            
            phishing_df = pd.DataFrame(phishing_data)
            print(phishing_df.to_string(index=False))
            phishing_df.to_csv('code/results/phishing_clustering_results.csv', index=False)
        
        # Find best models
        if 'emotion_data' in locals():
            best_emotion_model = emotion_df.loc[emotion_df['F1'].astype(float).idxmax()]
            print(f"\nBest emotion clustering model: {best_emotion_model['Model']}")
            print(f"   F1 Score: {best_emotion_model['F1']}")
            
        if 'phishing_data' in locals():
            best_phishing_model = phishing_df.loc[phishing_df['F1'].astype(float).idxmax()]
            print(f"\nBest phishing detection model: {best_phishing_model['Model']}")
            print(f"   F1 Score: {best_phishing_model['F1']}")
        
        print(f"\n✓ Results saved to CSV files in 'code/results/' directory")
        
        return True
    
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("Starting Machine Learning Pipeline...")
        print("="*80)
        
        # Execute pipeline steps
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
                print(f"Pipeline failed at step: {step.__name__}")
                return False
        
        print("\n" + "="*80)
        print("✓ Pipeline completed successfully!")
        print("="*80)
        return True

def main():
    """Main function to run the ML pipeline"""
    # Initialize pipeline with your dataset path
    # CSV headers: id,text,emotion,prompt_type,label
    dataset_path = 'code/combined_dataset.csv'  # Change this to your actual file path
    
    pipeline = MLPipeline(dataset_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
