import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('code/CURRENT_ANNOTATED.csv')

# Count total entries
total_entries = len(df)
print(f"Total entries in dataset: {total_entries}")
print(f"=" * 60)

# Count entries with annotations vs. empty
annotated = df['label'].notna() & (df['label'] != '')
annotated_count = annotated.sum()
empty_count = total_entries - annotated_count

print(f"\nAnnotation Coverage:")
print(f"  Annotated entries: {annotated_count} ({annotated_count/total_entries*100:.2f}%)")
print(f"  Empty/Missing labels: {empty_count} ({empty_count/total_entries*100:.2f}%)")
print(f"=" * 60)

# Extract all individual emotions
all_emotions = []
for label in df['label'].dropna():
    if label and isinstance(label, str):
        emotions = [e.strip() for e in label.split('#')]
        all_emotions.extend(emotions)

# Count occurrences of each emotion
emotion_counts = Counter(all_emotions)

# Sort by frequency
sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\nEmotion Annotation Counts:")
print(f"  Total emotion annotations: {len(all_emotions)}")
print(f"  Unique emotions: {len(emotion_counts)}")
print(f"  Average emotions per annotated entry: {len(all_emotions)/annotated_count:.2f}")
print(f"\n" + "-" * 60)

print(f"\nBreakdown by Emotion Type:")
for emotion, count in sorted_emotions:
    percentage = (count / len(all_emotions)) * 100
    print(f"  {emotion:20s}: {count:5d} ({percentage:5.2f}%)")

print(f"=" * 60)

# Multi-label statistics
multi_label_counts = []
for label in df[annotated]['label']:
    if label and isinstance(label, str):
        emotion_count = len(label.split('#'))
        multi_label_counts.append(emotion_count)

if multi_label_counts:
    print(f"\nMulti-label Statistics:")
    print(f"  Min emotions per entry: {min(multi_label_counts)}")
    print(f"  Max emotions per entry: {max(multi_label_counts)}")
    print(f"  Average emotions per entry: {sum(multi_label_counts)/len(multi_label_counts):.2f}")
    print(f"  Median emotions per entry: {sorted(multi_label_counts)[len(multi_label_counts)//2]}")
    
    # Count distribution of number of emotions
    label_count_dist = Counter(multi_label_counts)
    print(f"\n  Distribution of emotion count per entry:")
    for num_emotions, count in sorted(label_count_dist.items()):
        print(f"    {num_emotions} emotion(s): {count} entries")

print(f"=" * 60)

# Visualize the results
if sorted_emotions:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top emotions bar chart
    emotions_list = [e[0] for e in sorted_emotions[:15]]  # Top 15
    counts_list = [e[1] for e in sorted_emotions[:15]]
    
    axes[0].barh(emotions_list, counts_list, color='skyblue')
    axes[0].set_xlabel('Count')
    axes[0].set_title(f'Top 15 Emotion Annotations (Total: {len(all_emotions)} annotations)')
    axes[0].invert_yaxis()
    
    # Add count labels on bars
    for i, (emotion, count) in enumerate(zip(emotions_list, counts_list)):
        axes[0].text(count, i, f' {count}', va='center')
    
    # Annotation coverage pie chart
    axes[1].pie([annotated_count, empty_count], 
                labels=[f'Annotated\n({annotated_count})', f'Not Annotated\n({empty_count})'],
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'],
                startangle=90)
    axes[1].set_title('Annotation Coverage')
    
    plt.tight_layout()
    plt.savefig('annotation_statistics.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'annotation_statistics.png'")
    plt.show()

print("\nAnalysis complete!")
