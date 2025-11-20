#!/usr/bin/env python3
"""
Replace placeholder text in brackets with realistic fake information
"""

import pandas as pd
import re
import random
from datetime import datetime, timedelta

# Fake data generators
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
    }
    
    # Apply replacements
    for pattern, replacement_func in replacements.items():
        # Use re.sub with a lambda to call the function for each match
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
            # For any other placeholder, return a generic replacement
            return f"Details Available"
    
    # Replace any remaining [anything] patterns
    text = re.sub(r'\[([^\]]+)\]', replace_generic, text)
    
    return text

def process_csv_file(input_file, output_file):
    """Process the CSV file and replace all placeholders"""
    print(f"Loading CSV file: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Process the text column
    print("Replacing placeholders with realistic fake data...")
    df['text'] = df['text'].apply(replace_placeholders)
    
    # Save the processed file
    df.to_csv(output_file, index=False)
    print(f"✓ Processed file saved as: {output_file}")
    
    # Show some statistics
    print(f"\nProcessing complete:")
    print(f"- Input file: {input_file}")
    print(f"- Output file: {output_file}")
    print(f"- Rows processed: {len(df)}")

if __name__ == "__main__":
    input_file = "code/generated_10k_nonphishing_emails.csv"
    output_file = "code/generated_10k_nonphishing_emails_processed.csv"
    
    process_csv_file(input_file, output_file)