#!/usr/bin/env python3
"""compare.py [houses.csv] [datasets/dataset_truth.csv] - accuracy of a prediction file."""
import sys

from dslr_common import HOUSE_COLUMN, load_csv


def main():
	pred_file = sys.argv[1] if len(sys.argv) > 1 else 'houses.csv'
	truth_file = sys.argv[2] if len(sys.argv) > 2 else 'datasets/dataset_truth.csv'
	pred = load_csv(pred_file).set_index('Index')[HOUSE_COLUMN]
	truth = load_csv(truth_file).set_index('Index')[HOUSE_COLUMN]
	if len(pred) != len(truth) or not pred.index.equals(truth.index):
		sys.exit("Error: prediction and truth files do not cover the same Index values.")
	correct = int((pred == truth).sum())
	print(f"Accuracy: {correct / len(truth) * 100:.2f}% ({correct}/{len(truth)})")


if __name__ == "__main__":
	main()
