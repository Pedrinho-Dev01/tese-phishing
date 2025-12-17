import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Read the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'phishing_reports.csv')
df = pd.read_csv(csv_path)

# Convert Quarter column to datetime for proper plotting
def quarter_to_date(quarter_str):
    year, quarter = quarter_str.split('-Q')
    month = (int(quarter) - 1) * 3 + 1
    return datetime(int(year), month, 1)

df['Date'] = df['Quarter'].apply(quarter_to_date)

# Create the figure
plt.figure(figsize=(14, 8))
plt.plot(df['Date'], df['Phishing_Attacks_Detected'], 
         linewidth=2.5, color='#d62728', marker='o', markersize=4, label='Phishing Attacks Detected')

# Formatting
plt.title('Number of Phishing Attacks over Time (2013-2025)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Number of Attacks Detected', fontsize=12, fontweight='bold')

# Format x-axis to show only Q1 and Q3
ax = plt.gca()
# Filter for Q1 and Q3 only
tick_positions = [i for i in range(len(df)) if 'Q1' in df['Quarter'].iloc[i] or 'Q3' in df['Quarter'].iloc[i]]
tick_labels = [df['Quarter'].iloc[i] for i in tick_positions]
plt.xticks(df['Date'].iloc[tick_positions], tick_labels, rotation=45, ha='right', fontsize=10)

# Format y-axis with thousand separators
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Add legend
plt.legend(loc='upper left', fontsize=11)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), '..', 'latex', 'figs', 'phishing_trends.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to: {output_path}")
plt.close()
