#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def parse_labels(raw_label):
	"""Split a label field into individual normalized emotion labels."""
	if raw_label is None:
		return []

	if isinstance(raw_label, str):
		parts = raw_label.split("#")
	elif isinstance(raw_label, list):
		parts = raw_label
	else:
		return []

	return [part.strip().lower() for part in parts if str(part).strip()]


def count_emotions(dataset_path):
	with dataset_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		raise ValueError("Dataset root must be a JSON array.")

	counts = Counter()
	missing_label_rows = 0

	for row in data:
		if not isinstance(row, dict):
			continue

		labels = parse_labels(row.get("label"))
		if not labels:
			missing_label_rows += 1
			continue

		counts.update(labels)

	return counts, len(data), missing_label_rows


def save_counts_csv(counts, output_csv):
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["emotion", "annotation_count"])
		for emotion, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
			writer.writerow([emotion, count])


def plot_counts(counts, output_image):
	if not counts:
		raise ValueError("No emotion annotations found in dataset.")

	sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
	emotions = [item[0] for item in sorted_items]
	values = [item[1] for item in sorted_items]

	fig_width = max(10, len(emotions) * 0.5)
	plt.figure(figsize=(fig_width, 6))
	bars = plt.bar(emotions, values)

	plt.title("Emotion Annotation Counts")
	plt.xlabel("Emotion")
	plt.ylabel("Number of Annotations")
	plt.xticks(rotation=45, ha="right")

	for bar, value in zip(bars, values):
		plt.text(
			bar.get_x() + bar.get_width() / 2,
			value,
			str(value),
			ha="center",
			va="bottom",
			fontsize=8,
		)

	plt.tight_layout()
	output_image.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_image, dpi=180)
	plt.close()


def build_parser():
	parser = argparse.ArgumentParser(
		description="Count emotion annotations in dataset and generate a bar chart."
	)
	parser.add_argument(
		"--dataset",
		type=Path,
		default=Path(__file__).with_name("current_dataset.json"),
		help="Path to dataset JSON file (default: code/current_dataset.json).",
	)
	parser.add_argument(
		"--output-image",
		type=Path,
		default=Path(__file__).with_name("emotion_annotation_counts.png"),
		help="Path for output chart image.",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=Path(__file__).with_name("emotion_annotation_counts.csv"),
		help="Path for output CSV with counts.",
	)
	return parser


def main():
	args = build_parser().parse_args()

	counts, total_rows, missing_label_rows = count_emotions(args.dataset)
	save_counts_csv(counts, args.output_csv)
	plot_counts(counts, args.output_image)

	total_annotations = sum(counts.values())
	print(f"Rows in dataset: {total_rows}")
	print(f"Rows with missing/empty label: {missing_label_rows}")
	print(f"Total emotion annotations counted: {total_annotations}")
	print(f"Unique emotions: {len(counts)}")
	print(f"Saved chart: {args.output_image}")
	print(f"Saved counts CSV: {args.output_csv}")


if __name__ == "__main__":
	main()
