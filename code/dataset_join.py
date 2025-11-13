import pandas as pd
from sklearn.utils import shuffle
# Load the two datasets
df_nonphishing = pd.read_csv('code/generated_10k_nonphishing_emails.csv')
df_phishing = pd.read_csv('code/generated_10k_emails.csv')
# Shuffle both datasets
df_nonphishing = shuffle(df_nonphishing, random_state=42).reset_index(drop=True)
df_phishing = shuffle(df_phishing, random_state=42).reset_index(drop=True)
# Assume both datasets are the same size
min_length = min(len(df_nonphishing), len(df_phishing))
df_nonphishing = df_nonphishing.iloc[:min_length]
df_phishing = df_phishing.iloc[:min_length]
# Add a new column to label the datasets
df_nonphishing['label'] = 'non-phishing'
df_phishing['label'] = 'phishing'
# Combine the datasets
df_combined = pd.concat([df_nonphishing, df_phishing], ignore_index=True)
# Shuffle the combined dataset
df_combined = shuffle(df_combined, random_state=42).reset_index(drop=True)
# Save the combined dataset
df_combined.to_csv('code/combined_emails_dataset.csv', index=False)
