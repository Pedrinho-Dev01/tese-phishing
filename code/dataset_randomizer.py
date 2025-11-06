import json
import random

# Load your dataset with UTF-8 encoding
with open('code/generated_10k_emails.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Randomize the order of entries
random.shuffle(data)

# Save the randomized dataset with UTF-8 encoding
with open('dataset_randomized.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Successfully randomized {len(data)} entries!")
print("Saved to: dataset_randomized.json")