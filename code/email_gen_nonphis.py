# Generate 10,000 phishing emails distributed evenly across all emotions
# Based on examples from base_annotations.csv

import pandas as pd
from ollama import Client

# Configuration
ORIGINAL_EMAILS = 9000
SECOND_PROMPT_EMAILS = 1000
TOTAL_EMAILS = ORIGINAL_EMAILS + SECOND_PROMPT_EMAILS

# Load the CSV data
data = pd.read_csv('code/base_annotations.csv')
grouped_emails = data.groupby('emotion')['text'].apply(list).to_dict()

# Calculate emails per emotion for original prompt
num_emotions = len(grouped_emails)
original_per_emotion = ORIGINAL_EMAILS // num_emotions
original_remaining = ORIGINAL_EMAILS % num_emotions

# Calculate emails per emotion for second prompt
second_per_emotion = SECOND_PROMPT_EMAILS // num_emotions
second_remaining = SECOND_PROMPT_EMAILS % num_emotions

print(f"Generating {TOTAL_EMAILS} emails total:")
print(f"  - {ORIGINAL_EMAILS} with original prompt across {num_emotions} emotions")
print(f"  - {SECOND_PROMPT_EMAILS} with second prompt across {num_emotions} emotions")
print(f"\nOriginal prompt distribution:")
print(f"  Base: {original_per_emotion} emails per emotion")
if original_remaining > 0:
    print(f"  Extra: {original_remaining} additional emails for first {original_remaining} emotions")
print(f"\nSecond prompt distribution:")
print(f"  Base: {second_per_emotion} emails per emotion")
if second_remaining > 0:
    print(f"  Extra: {second_remaining} additional emails for first {second_remaining} emotions")

# Initialize Ollama client
client = Client(
    host='http://192.168.238.77:8001/',
    headers={}
)

# Define second prompt template (you can modify this)
SECOND_PROMPT_TEMPLATE = """Here are example emails that express the emotion '{emotion}':

{context}

Based on these examples, generate 1 new email with a different approach or style while still maintaining the '{emotion}' emotion. 

===============================

Subject: Thesis progress update and meeting scheduling

Good morning,

I hope you’re doing well.
I’m sending the slides I intend to use for the upcoming thesis progress presentation. I would also like to confirm whether a meeting will be needed this week or if we could schedule a short one at a convenient time.

Best regards,
[Your name]

===============================

Subject: Request for access to file / resource

Good afternoon,

Could you please grant me access to the file [file name] or let me know where I can find it?
I need it to continue my thesis work.

Thank you in advance for your help.

Best regards,
[Your name]

===============================

Subject: Meeting scheduling

Good morning,

I’d like to ask if it would be possible to meet sometime between [time X] and [time Y].
If that’s not convenient, I can adjust to another time this week.

Best regards,
[Your name]

===============================

Subject: Follow-up on previous meeting

Good evening,

As discussed last week, progress currently depends on the [annotators / data / results].
Once there are updates, I can send a summary or schedule a new meeting.

Best regards,
[Your name]

===============================

Generate only the email content, no additional text or explanations:"""

# Generate emails
all_generated_emails = []

print("\n" + "="*60)
print("PHASE 1: Generating emails with ORIGINAL prompt")
print("="*60)

emotion_count = 0
for emotion, example_emails in grouped_emails.items():
    emotion_count += 1
    
    # Calculate how many emails to generate for this emotion (original prompt)
    emails_to_generate = original_per_emotion
    if emotion_count <= original_remaining:
        emails_to_generate += 1
    
    print(f"Generating {emails_to_generate} emails for '{emotion}' ({emotion_count}/{num_emotions})")
    
    # Create context with examples from this emotion
    context = f"Here are example emails that express the emotion '{emotion}':\n\n"
    for i, email in enumerate(example_emails[:5], 1):
        context += f"Example {i}: {email}\n\n"
    
    # Generate the required number of emails for this emotion (original prompt)
    for i in range(emails_to_generate):
        prompt = f"""{context}

Based on these examples of phishing emails expressing '{emotion}', generate 1 new realistic non-phishing email that captures the same emotional tone and style. Make it convincing and professional, evoking the emotion of {emotion}.

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
                'emotion': emotion,
                'prompt_type': 'original'
            })
            
            if (i + 1) % 10 == 0 or i == emails_to_generate - 1:
                print(f"  Progress: {i + 1}/{emails_to_generate}")
                
        except Exception as e:
            print(f"  Error generating email {i + 1}: {str(e)}")

print("\n" + "="*60)
print("PHASE 2: Generating emails with SECOND prompt")
print("="*60)

emotion_count = 0
for emotion, example_emails in grouped_emails.items():
    emotion_count += 1
    
    # Calculate how many emails to generate for this emotion (second prompt)
    emails_to_generate = second_per_emotion
    if emotion_count <= second_remaining:
        emails_to_generate += 1
    
    print(f"Generating {emails_to_generate} emails for '{emotion}' with second prompt ({emotion_count}/{num_emotions})")
    
    # Create context with examples from this emotion
    context_examples = ""
    for i, email in enumerate(example_emails[:5], 1):
        context_examples += f"Example {i}: {email}\n\n"
    
    # Generate the required number of emails for this emotion (second prompt)
    for i in range(emails_to_generate):
        prompt = SECOND_PROMPT_TEMPLATE.format(
            emotion=emotion,
            context=context_examples
        )
        
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
                'emotion': emotion,
                'prompt_type': 'second'
            })
            
            if (i + 1) % 10 == 0 or i == emails_to_generate - 1:
                print(f"  Progress: {i + 1}/{emails_to_generate}")
                
        except Exception as e:
            print(f"  Error generating email {i + 1}: {str(e)}")

# Save all generated emails to CSV
output_df = pd.DataFrame(all_generated_emails)
output_df.to_csv('code/generated_10k_nonphishing_emails.csv', index=False)

print(f"\n" + "="*60)
print(f"COMPLETED! Generated {len(all_generated_emails)} emails")
print(f"Results saved to 'code/generated_10k_nonphishing_emails.csv'")
print("="*60)

# Show distribution summary
emotion_counts = output_df['emotion'].value_counts()
prompt_type_counts = output_df['prompt_type'].value_counts()

print(f"\nDistribution by emotion:")
for emotion, count in emotion_counts.items():
    print(f"  {emotion}: {count} emails")

print(f"\nDistribution by prompt type:")
for prompt_type, count in prompt_type_counts.items():
    print(f"  {prompt_type}: {count} emails")

# Show detailed breakdown
print(f"\nDetailed breakdown:")
breakdown = output_df.groupby(['emotion', 'prompt_type']).size().unstack(fill_value=0)
print(breakdown)