#!/usr/bin/env python3
"""
Script to remove prompt_type column from a dataset
"""

import pandas as pd
import os

def remove_prompt_type_column(input_file):
    """
    Remove prompt_type column from a CSV dataset (overwrites original file)
    
    Args:
        input_file (str): Path to input CSV file
    
    Returns:
        bool: Success status
    """
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"✗ Error: File '{input_file}' not found")
            return False
        
        print(f"Loading dataset: {input_file}")
        
        # Load the dataset
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Check if prompt_type column exists
        if 'prompt_type' not in df.columns:
            print("✓ No 'prompt_type' column found - nothing to remove")
            return True
        
        # Show prompt_type distribution before removal
        print(f"\nPrompt_type distribution:")
        print(df['prompt_type'].value_counts())
        
        # Remove prompt_type column
        df_cleaned = df.drop('prompt_type', axis=1)
        
        print(f"\nAfter removal:")
        print(f"New shape: {df_cleaned.shape}")
        print(f"New columns: {list(df_cleaned.columns)}")
        
        # Overwrite original file
        df_cleaned.to_csv(input_file, index=False)
        
        print(f"\n✓ File updated: {input_file}")
        print(f"✓ Removed 1 column, kept {len(df_cleaned.columns)} columns")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        return False

def main():
    """Main function - automatically processes all found files"""
    
    # Default files to try
    default_files = [
        'code/combined_emails_dataset.csv',
        'code/generated_10k_emails.csv', 
        'code/generated_10k_nonphishing_emails.csv',
        'code/combined_dataset.csv'
    ]
    
    print("Dataset Prompt_Type Column Remover")
    print("="*50)
    print("Automatically processing all files with prompt_type column...\n")
    
    files_processed = 0
    files_with_prompt_type = 0
    
    for file_path in default_files:
        if os.path.exists(file_path):
            try:
                # Quick check for prompt_type column
                df_temp = pd.read_csv(file_path)
                if 'prompt_type' in df_temp.columns:
                    files_with_prompt_type += 1
                    print(f"Processing: {file_path} ({df_temp.shape[0]:,} rows)")
                    success = remove_prompt_type_column(file_path)
                    if success:
                        files_processed += 1
                    print("-" * 60)
                else:
                    print(f"Skipped: {file_path} (no prompt_type column)")
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
        else:
            print(f"Not found: {file_path}")
    
    print(f"\nSummary:")
    print(f"✓ Files processed: {files_processed}/{files_with_prompt_type}")
    print(f"Files with prompt_type found: {files_with_prompt_type}")
    
    if files_processed > 0:
        print("\n✓ All operations completed successfully!")
    else:
        print("\n! No files needed processing")

if __name__ == "__main__":
    main()