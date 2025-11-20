import pandas as pd
from sklearn.utils import shuffle

# Load the two datasets
df_nonphishing = pd.read_csv('code/generated_10k_nonphishing_emails.csv')
df_phishing = pd.read_csv('code/base_annotations.csv')

print(f"Original dataset sizes:")
print(f"Non-phishing: {len(df_nonphishing):,} entries")
print(f"Phishing: {len(df_phishing):,} entries")

# Shuffle both datasets
df_nonphishing = shuffle(df_nonphishing, random_state=42).reset_index(drop=True)
df_phishing = shuffle(df_phishing, random_state=42).reset_index(drop=True)

# Balance datasets by sampling from larger dataset to match smaller one
min_length = min(len(df_nonphishing), len(df_phishing))
max_length = max(len(df_nonphishing), len(df_phishing))

print(f"\nBalancing datasets to {min_length:,} entries each")

# Sample from larger dataset to match smaller one
if len(df_nonphishing) > len(df_phishing):
    df_nonphishing = df_nonphishing.sample(n=min_length, random_state=42).reset_index(drop=True)
    print(f"Sampled {min_length:,} from {max_length:,} non-phishing entries")
elif len(df_phishing) > len(df_nonphishing):
    df_phishing = df_phishing.sample(n=min_length, random_state=42).reset_index(drop=True)
    print(f"Sampled {min_length:,} from {max_length:,} phishing entries")
else:
    print("Datasets already balanced")

# Add labels to identify dataset origin
df_nonphishing['label'] = 'non-phishing'
df_phishing['label'] = 'phishing'

# Combine the balanced datasets
df_combined = pd.concat([df_nonphishing, df_phishing], ignore_index=True)

# Final shuffle of combined dataset
df_combined = shuffle(df_combined, random_state=42).reset_index(drop=True)

print(f"\nFinal combined dataset:")
print(f"Total entries: {len(df_combined):,}")
print(f"Label distribution:")
print(df_combined['label'].value_counts())

# Save the balanced combined dataset
df_combined.to_csv('code/combined_emails_dataset.csv', index=False)
print(f"\n✓ Balanced dataset saved to 'code/combined_emails_dataset.csv'")
