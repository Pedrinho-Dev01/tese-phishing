#!/usr/bin/env python3
"""
Complete Dataset Processing Pipeline
Handles placeholder replacement, column removal, and dataset balancing/joining
"""

import pandas as pd
import re
import random
import os
import numpy as np
from sklearn.utils import shuffle
from datetime import datetime, timedelta

# ============================================================================
# PLACEHOLDER REPLACEMENT DATA
# ============================================================================

RECIPIENTS = [
    "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis", "David Wilson",
    "Jessica Martinez", "Daniel Garcia", "Ashley Rodriguez", "Christopher Lee", "Amanda Taylor",
    "Matthew Anderson", "Jennifer Thomas", "Joshua Jackson", "Elizabeth White", "Andrew Harris",
    "Stephanie Martin", "Ryan Thompson", "Nicole Garcia", "Brandon Clark", "Samantha Lewis",
    "Kevin Robinson", "Rachel Walker", "Tyler Hall", "Lauren Allen", "Justin Young",
    "Megan Hernandez", "Nicholas King", "Kayla Wright", "Zachary Lopez", "Victoria Hill",
    "Alexander Scott", "Brittany Green", "Jonathan Adams", "Danielle Baker", "Nathan Gonzalez"
]

SENDER_NAMES = [
    "Mark Stevens", "Lisa Chen", "Robert Johnson", "Maria Rodriguez", "James Wilson",
    "Anna Thompson", "David Miller", "Susan Davis", "Michael Garcia", "Jennifer Brown",
    "Christopher Lee", "Karen Martinez", "Daniel Anderson", "Patricia Taylor", "Matthew Thomas",
    "Linda Jackson", "Anthony White", "Barbara Harris", "Joshua Martin", "Elizabeth Clark",
    "Andrew Lewis", "Mary Robinson", "Ryan Walker", "Nancy Hall", "Kevin Allen"
]

POSITIONS = [
    "Customer Success Manager", "Marketing Director", "Operations Manager", "Community Manager",
    "Senior Account Executive", "Business Development Manager", "Customer Relations Specialist",
    "Program Manager", "Regional Director", "VP of Customer Experience", "Head of Marketing",
    "Senior Consultant", "Team Lead", "Project Coordinator", "Account Manager"
]

COMPANIES = [
    "TechSolutions Inc", "GlobalCorp", "InnovateTech", "FutureSoft", "DataDyne Systems",
    "CloudNine Technologies", "NextGen Solutions", "SmartBusiness Co", "DigiCorp",
    "TechForward", "InnovateNow", "BusinessPro Inc", "TechAdvantage", "ModernSoft",
    "DigitalFirst Corp", "TechExcellence", "BusinessEdge", "SmartTech Solutions"
]

SUBSCRIPTION_PLANS = [
    "Basic Plan", "Standard Plan", "Premium Plan", "Professional Plan", "Enterprise Plan",
    "Starter Package", "Advanced Package", "Business Suite", "Pro Membership", "Elite Subscription"
]

FIELDS_INDUSTRIES = [
    "technology", "healthcare", "education", "environmental conservation", "business development",
    "marketing", "finance", "research and development", "customer service", "project management",
    "data analysis", "software development", "digital marketing", "consulting", "non-profit work"
]

CONTACT_INFO = [
    "support@company.com | (555) 123-4567", "info@business.com | (555) 987-6543",
    "contact@techsolutions.com | (555) 456-7890", "hello@innovate.com | (555) 234-5678",
    "customercare@global.com | (555) 345-6789", "team@nextgen.com | (555) 567-8901"
]

SUSPICIOUS_LINKS = [
    "http://secure-login.com/verify", "http://account-update.com/login",
    "http://payment-confirmation.com/secure", "http://urgent-action-required.com",
    "http://verify-your-account.com", "http://security-check.com/login"
]

# ============================================================================
# PLACEHOLDER REPLACEMENT FUNCTIONS
# ============================================================================

def generate_random_date(days_ahead=30):
    """Generate a random date within the next X days"""
    future_date = datetime.now() + timedelta(days=random.randint(1, days_ahead))
    return future_date.strftime("%B %d, %Y")

def generate_random_time():
    """Generate a random time"""
    hour = random.randint(9, 17)  # Business hours
    minute = random.choice([0, 15, 30, 45])
    return f"{hour:02d}:{minute:02d}"

def generate_random_duration():
    """Generate random duration"""
    durations = ["6 months", "1 year", "90 days", "3 months", "12 months", "180 days"]
    return random.choice(durations)

def generate_random_rsvp_date():
    """Generate RSVP date (usually sooner than event date)"""
    rsvp_date = datetime.now() + timedelta(days=random.randint(3, 14))
    return rsvp_date.strftime("%B %d, %Y")

def replace_placeholders(text):
    """Replace all placeholders in brackets with realistic fake data"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Replace specific placeholders with realistic data
    replacements = {
        r'\[Recipient\]': lambda: random.choice(RECIPIENTS),
        r'\[Your Name\]': lambda: random.choice(SENDER_NAMES),
        r'\[Your Position\]': lambda: random.choice(POSITIONS),
        r'\[Company Name\]': lambda: random.choice(COMPANIES),
        r'\[Organization Name\]': lambda: random.choice(COMPANIES),
        r'\[Date\]': lambda: generate_random_date(),
        r'\[Time\]': lambda: generate_random_time(),
        r'\[Duration of the upgrade\]': lambda: generate_random_duration(),
        r'\[Duration\]': lambda: generate_random_duration(),
        r'\[Reply Deadline\]': lambda: generate_random_rsvp_date(),
        r'\[RSVP Date\]': lambda: generate_random_rsvp_date(),
        r'\[Contact Information\]': lambda: random.choice(CONTACT_INFO),
        r'\[Your Current Plan\]': lambda: random.choice(SUBSCRIPTION_PLANS[:3]),  # Lower tier plans
        r'\[Proposed Upgrade Plan\]': lambda: random.choice(SUBSCRIPTION_PLANS[2:]),  # Higher tier plans
        r'\[field/industry\]': lambda: random.choice(FIELDS_INDUSTRIES),
        r'\[field\]': lambda: random.choice(FIELDS_INDUSTRIES),
        r'\[industry\]': lambda: random.choice(FIELDS_INDUSTRIES),
        r'\[Suspicious Link\]': lambda: random.choice(SUSPICIOUS_LINKS),
    }
    
    # Apply replacements
    for pattern, replacement_func in replacements.items():
        text = re.sub(pattern, lambda m: replacement_func(), text, flags=re.IGNORECASE)
    
    # Handle any remaining generic placeholders
    def replace_generic(match):
        placeholder = match.group(1).lower()
        if 'name' in placeholder:
            return random.choice(SENDER_NAMES)
        elif 'date' in placeholder:
            return generate_random_date()
        elif 'time' in placeholder:
            return generate_random_time()
        elif 'company' in placeholder or 'organization' in placeholder:
            return random.choice(COMPANIES)
        elif 'plan' in placeholder:
            return random.choice(SUBSCRIPTION_PLANS)
        elif 'position' in placeholder or 'title' in placeholder:
            return random.choice(POSITIONS)
        elif 'contact' in placeholder or 'email' in placeholder or 'phone' in placeholder:
            return random.choice(CONTACT_INFO)
        else:
            return f"Details Available"
    
    # Replace any remaining [anything] patterns
    text = re.sub(r'\[([^\]]+)\]', replace_generic, text)
    
    return text

# ============================================================================
# COLUMN REMOVAL FUNCTIONS
# ============================================================================

def remove_prompt_type_column(df, filename="dataset"):
    """Remove prompt_type column from a dataframe"""
    if 'prompt_type' not in df.columns:
        print(f"✓ {filename}: No 'prompt_type' column found - nothing to remove")
        return df
    
    print(f"{filename}: Removing 'prompt_type' column...")
    print(f"  Original columns: {list(df.columns)}")
    print(f"  Prompt_type distribution: {df['prompt_type'].value_counts().to_dict()}")
    
    df_cleaned = df.drop('prompt_type', axis=1)
    print(f"  New columns: {list(df_cleaned.columns)}")
    print(f"  ✓ Removed 1 column, kept {len(df_cleaned.columns)} columns")
    
    return df_cleaned

# ============================================================================
# DATASET BALANCING FUNCTIONS
# ============================================================================

def balance_emotions_in_phishing(df_phishing, target_samples_per_emotion):
    """Balance emotions within phishing dataset"""
    print(f"\nBalancing emotions to {target_samples_per_emotion} samples each...")
    
    emotion_counts = df_phishing['emotion'].value_counts().sort_index()
    print("Original emotion distribution:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")
    
    balanced_samples = []
    
    for emotion in sorted(df_phishing['emotion'].unique()):
        emotion_samples = df_phishing[df_phishing['emotion'] == emotion]
        
        if len(emotion_samples) >= target_samples_per_emotion:
            # Sample exactly target_samples_per_emotion
            balanced_sample = emotion_samples.sample(n=target_samples_per_emotion, random_state=42)
            balanced_samples.append(balanced_sample)
            print(f"  {emotion}: sampled {target_samples_per_emotion} from {len(emotion_samples)} available")
        else:
            # Not enough samples for this emotion
            print(f"  WARNING: {emotion} has only {len(emotion_samples)} samples, need {target_samples_per_emotion}")
            balanced_samples.append(emotion_samples)
    
    df_balanced = pd.concat(balanced_samples, ignore_index=True)
    df_balanced = shuffle(df_balanced, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced phishing dataset: {len(df_balanced):,} entries")
    print("Final emotion distribution:")
    final_counts = df_balanced['emotion'].value_counts().sort_index()
    for emotion, count in final_counts.items():
        print(f"  {emotion}: {count}")
    
    return df_balanced

# ============================================================================
# MAIN PROCESSING CLASS
# ============================================================================

class DatasetProcessor:
    def __init__(self):
        """Initialize the dataset processor"""
        self.df_nonphishing = None
        self.df_phishing = None
        self.df_combined = None
        
    def load_datasets(self, nonphishing_file='code/generated_10k_nonphishing_emails.csv', 
                     phishing_file='code/base_annotations.csv'):
        """Load the raw datasets"""
        print("=" * 80)
        print("DATASET PROCESSOR - Loading Datasets")
        print("=" * 80)
        
        # Load non-phishing dataset
        if os.path.exists(nonphishing_file):
            self.df_nonphishing = pd.read_csv(nonphishing_file)
            print(f"✓ Loaded non-phishing: {nonphishing_file} ({len(self.df_nonphishing):,} entries)")
        else:
            print(f"✗ Non-phishing file not found: {nonphishing_file}")
            return False
            
        # Load phishing dataset
        if os.path.exists(phishing_file):
            self.df_phishing = pd.read_csv(phishing_file)
            print(f"✓ Loaded phishing: {phishing_file} ({len(self.df_phishing):,} entries)")
        else:
            print(f"✗ Phishing file not found: {phishing_file}")
            return False
            
        print(f"\nDataset columns:")
        print(f"  Non-phishing: {list(self.df_nonphishing.columns)}")
        print(f"  Phishing: {list(self.df_phishing.columns)}")
        
        return True
        
    def process_placeholders(self):
        """Replace placeholders in text columns"""
        print("\n" + "=" * 80)
        print("STEP 1: REPLACING PLACEHOLDERS")
        print("=" * 80)
        
        # Process non-phishing dataset
        if 'text' in self.df_nonphishing.columns:
            print("Processing non-phishing placeholders...")
            original_sample = self.df_nonphishing['text'].iloc[0][:100] + "..."
            print(f"  Original sample: {original_sample}")
            
            self.df_nonphishing['text'] = self.df_nonphishing['text'].apply(replace_placeholders)
            
            processed_sample = self.df_nonphishing['text'].iloc[0][:100] + "..."
            print(f"  Processed sample: {processed_sample}")
            print(f"  ✓ Processed {len(self.df_nonphishing)} non-phishing texts")
        
        # Process phishing dataset  
        if 'text' in self.df_phishing.columns:
            print("\nProcessing phishing placeholders...")
            original_sample = self.df_phishing['text'].iloc[0][:100] + "..."
            print(f"  Original sample: {original_sample}")
            
            self.df_phishing['text'] = self.df_phishing['text'].apply(replace_placeholders)
            
            processed_sample = self.df_phishing['text'].iloc[0][:100] + "..."
            print(f"  Processed sample: {processed_sample}")
            print(f"  ✓ Processed {len(self.df_phishing)} phishing texts")
        
        return True
        
    def remove_unwanted_columns(self):
        """Remove prompt_type columns if they exist"""
        print("\n" + "=" * 80)
        print("STEP 2: REMOVING UNWANTED COLUMNS")
        print("=" * 80)
        
        # Remove from non-phishing
        self.df_nonphishing = remove_prompt_type_column(self.df_nonphishing, "Non-phishing")
        
        # Remove from phishing
        self.df_phishing = remove_prompt_type_column(self.df_phishing, "Phishing")
        
        return True
        
    def balance_and_combine_datasets(self):
        """Balance emotions and combine datasets"""
        print("\n" + "=" * 80)
        print("STEP 3: BALANCING AND COMBINING DATASETS")
        print("=" * 80)
        
        print(f"Original dataset sizes:")
        print(f"  Non-phishing: {len(self.df_nonphishing):,} entries")
        print(f"  Phishing: {len(self.df_phishing):,} entries")
        
        # Shuffle both datasets
        self.df_nonphishing = shuffle(self.df_nonphishing, random_state=42).reset_index(drop=True)
        self.df_phishing = shuffle(self.df_phishing, random_state=42).reset_index(drop=True)
        
        # Determine balancing strategy
        min_dataset_size = min(len(self.df_nonphishing), len(self.df_phishing))
        num_emotions = self.df_phishing['emotion'].nunique()
        samples_per_emotion = min_dataset_size // num_emotions
        
        print(f"\nBalancing strategy:")
        print(f"  Smaller dataset size: {min_dataset_size:,}")
        print(f"  Number of unique emotions: {num_emotions}")
        print(f"  Samples per emotion: {samples_per_emotion}")
        
        # Balance emotions within phishing dataset
        self.df_phishing_balanced = balance_emotions_in_phishing(self.df_phishing, samples_per_emotion)
        
        # Balance non-phishing to match phishing size
        target_size = len(self.df_phishing_balanced)
        print(f"\nBalancing non-phishing to {target_size:,} entries")
        
        if len(self.df_nonphishing) > target_size:
            self.df_nonphishing = self.df_nonphishing.sample(n=target_size, random_state=42).reset_index(drop=True)
            print(f"  ✓ Sampled {target_size:,} from original non-phishing entries")
        elif len(self.df_nonphishing) < target_size:
            print(f"  WARNING: Non-phishing has only {len(self.df_nonphishing):,} entries, need {target_size:,}")
        else:
            print("  ✓ Non-phishing dataset already matches target size")
        
        # Add labels
        print(f"\nAdding dataset labels...")
        self.df_nonphishing['label'] = 'non-phishing'
        self.df_phishing_balanced['label'] = 'phishing'
        
        # Combine datasets
        self.df_combined = pd.concat([self.df_nonphishing, self.df_phishing_balanced], ignore_index=True)
        self.df_combined = shuffle(self.df_combined, random_state=42).reset_index(drop=True)
        
        print(f"\nFinal combined dataset:")
        print(f"  Total entries: {len(self.df_combined):,}")
        print(f"  Label distribution: {self.df_combined['label'].value_counts().to_dict()}")
        
        # Show final emotion distribution for phishing emails
        phishing_subset = self.df_combined[self.df_combined['label'] == 'phishing']
        if 'emotion' in phishing_subset.columns:
            print(f"\n  Final emotion distribution in phishing emails:")
            emotion_dist = phishing_subset['emotion'].value_counts().sort_index()
            for emotion, count in emotion_dist.items():
                print(f"    {emotion}: {count}")
        
        return True
        
    def save_processed_dataset(self, output_file='code/combined_emails_dataset.csv'):
        """Save the final processed dataset"""
        print("\n" + "=" * 80)
        print("STEP 4: SAVING PROCESSED DATASET")
        print("=" * 80)
        
        if self.df_combined is None:
            print("✗ No combined dataset to save")
            return False
            
        # Save the dataset
        self.df_combined.to_csv(output_file, index=False)
        print(f"✓ Dataset saved to: {output_file}")
        
        # Final statistics
        print(f"\nFinal Statistics:")
        print(f"  Total rows: {len(self.df_combined):,}")
        print(f"  Columns: {list(self.df_combined.columns)}")
        print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        # Dataset breakdown
        phishing_count = len(self.df_combined[self.df_combined['label'] == 'phishing'])
        nonphishing_count = len(self.df_combined[self.df_combined['label'] == 'non-phishing'])
        
        if 'emotion' in self.df_combined.columns:
            emotions_count = len(self.df_combined[self.df_combined['label'] == 'phishing']['emotion'].unique())
            samples_per_emotion = phishing_count // emotions_count
            
            print(f"  Phishing emails: {phishing_count:,} ({emotions_count} emotions × ~{samples_per_emotion})")
        else:
            print(f"  Phishing emails: {phishing_count:,}")
            
        print(f"  Non-phishing emails: {nonphishing_count:,}")
        
        return True
        
    def run_complete_pipeline(self, 
                            nonphishing_file='code/generated_10k_nonphishing_emails.csv',
                            phishing_file='code/base_annotations.csv',
                            output_file='code/combined_emails_dataset.csv'):
        """Run the complete dataset processing pipeline"""
        print("STARTING COMPLETE DATASET PROCESSING PIPELINE")
        print("=" * 80)
        
        steps = [
            ("Loading datasets", lambda: self.load_datasets(nonphishing_file, phishing_file)),
            ("Processing placeholders", self.process_placeholders),
            ("Removing unwanted columns", self.remove_unwanted_columns),
            ("Balancing and combining", self.balance_and_combine_datasets),
            ("Saving final dataset", lambda: self.save_processed_dataset(output_file))
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"✗ Pipeline failed at: {step_name}")
                return False
        
        print("\n" + "=" * 80)
        print("✓ DATASET PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final dataset available at: {output_file}")
        print(f"Dataset ready for machine learning pipeline!")
        
        return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete dataset processing pipeline"""
    processor = DatasetProcessor()
    
    # Run the complete pipeline
    success = processor.run_complete_pipeline(
        nonphishing_file='code/generated_10k_nonphishing_emails.csv',
        phishing_file='code/generated_10k_emails.csv',
        output_file='code/combined_emails_dataset.csv'
    )
    
    if success:
        print("\n✓ Ready to run model_pipeline.py with the processed dataset!")
    else:
        print("\n✗ Processing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()