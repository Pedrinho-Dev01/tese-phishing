# Generate 10,000 phishing emails distributed evenly across all emotions
# Based on examples from base_annotations.csv

import pandas as pd
from ollama import Client

# Configuration
TOTAL_EMAILS = 10000

# Load the CSV data
data = pd.read_csv('base_annotations.csv')
grouped_emails = data.groupby('emotion')['text'].apply(list).to_dict()

# Calculate emails per emotion
num_emotions = len(grouped_emails)
emails_per_emotion = TOTAL_EMAILS // num_emotions
remaining_emails = TOTAL_EMAILS % num_emotions

print(f"Generating {TOTAL_EMAILS} emails across {num_emotions} emotions")
print(f"Base: {emails_per_emotion} emails per emotion")
if remaining_emails > 0:
    print(f"Extra: {remaining_emails} additional emails for first {remaining_emails} emotions")

# Initialize Ollama client
client = Client(
    host='http://192.168.238.77:8000/',
    headers={}
)

# Generate emails
all_generated_emails = []
emotion_count = 0

for emotion, example_emails in grouped_emails.items():
    emotion_count += 1
    
    # Calculate how many emails to generate for this emotion
    emails_to_generate = emails_per_emotion
    if emotion_count <= remaining_emails:
        emails_to_generate += 1
    
    print(f"Generating {emails_to_generate} emails for '{emotion}' ({emotion_count}/{num_emotions})")
    
    # Create context with examples from this emotion
    context = f"Here are example emails that express the emotion '{emotion}':\n\n"
    for i, email in enumerate(example_emails[:5], 1):
        context += f"Example {i}: {email}\n\n"
    
    # Generate the required number of emails for this emotion
    for i in range(emails_to_generate):
        prompt = f"""{context}

Based on these examples of emails expressing '{emotion}', generate 1 new realistic phishing email that captures the same emotional tone and style. Make it convincing and professional, evoking the emotion of {emotion}.

Generate only the email content, no additional text or explanations:"""
        
        try:
            response = client.chat(model='dolphin3:8b', messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            generated_email = response['message']['content'].strip()
            all_generated_emails.append({
                'id': len(all_generated_emails) + 1,
                'text': generated_email,
                'emotion': emotion
            })
            
            if (i + 1) % 10 == 0 or i == emails_to_generate - 1:
                print(f"  Progress: {i + 1}/{emails_to_generate}")
                
        except Exception as e:
            print(f"  Error generating email {i + 1}: {str(e)}")

# Save all generated emails to CSV
output_df = pd.DataFrame(all_generated_emails)
output_df.to_csv('generated_10k_emails.csv', index=False)

print(f"\nCompleted! Generated {len(all_generated_emails)} emails")
print(f"Results saved to 'generated_10k_emails.csv'")

# Show distribution summary
emotion_counts = output_df['emotion'].value_counts()
print(f"\nDistribution summary:")
for emotion, count in emotion_counts.items():
    print(f"  {emotion}: {count} emails")