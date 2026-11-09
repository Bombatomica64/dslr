#!/usr/bin/env python3
"""pair_plot.py dataset_train.csv [-o out.png]

From this visualization, what features are you going to use for your logistic regression?
"""
import argparse
import textwrap

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from dslr_common import (
	DROPPED_FEATURES,
	FEATURES,
	HOUSE_COLORS,
	HOUSE_COLUMN,
	load_csv,
	numeric_courses,
	require_house_column,
)


def main() -> None:
	parser = argparse.ArgumentParser(description='Pair plot of every course, per house.')
	parser.add_argument('dataset')
	parser.add_argument('-o', '--output', help='save the figure instead of only showing it')
	args = parser.parse_args()

	data = load_csv(args.dataset)
	require_house_column(data, args.dataset)
	courses = numeric_courses(data)
	if len(courses) < 2:
		raise SystemExit("Error: need at least 2 numeric features.")
	houses = sorted(data[HOUSE_COLUMN].dropna().unique())

	print(f"Answer: features used for training ({len(FEATURES)}): {', '.join(FEATURES)}")
	print("Dropped:")
	for feature, reason in DROPPED_FEATURES.items():
		print(f"  {feature}: {reason}")

	n = len(courses)
	fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
	for i, row in enumerate(courses):
		for j, col in enumerate(courses):
			ax = axes[i, j]
			for house in houses:
				subset = data[data[HOUSE_COLUMN] == house]
				if i == j:
					ax.hist(subset[row].dropna(), bins=20, alpha=0.5, color=HOUSE_COLORS.get(house))
				else:
					ax.scatter(subset[col], subset[row], s=2, alpha=0.4, color=HOUSE_COLORS.get(house))
			ax.set_xticks([])
			ax.set_yticks([])
			if j == 0:
				ax.set_ylabel(textwrap.fill(row, 14), fontsize=8,
					color='black' if row in FEATURES else 'darkred')
			if i == n - 1:
				ax.set_xlabel(textwrap.fill(col, 14), fontsize=8,
					color='black' if col in FEATURES else 'darkred')
	handles = [mlines.Line2D([], [], marker='o', ls='', color=HOUSE_COLORS.get(h)) for h in houses]
	fig.legend(handles, houses, loc='upper right')
	fig.suptitle("Diagonal: per-house histograms. Features used for training (red labels = dropped):\n"
		+ textwrap.fill(', '.join(FEATURES), 120), fontsize=14)
	fig.tight_layout(rect=(0, 0, 0.93, 0.95))
	if args.output:
		fig.savefig(args.output, dpi=80)
		print(f"Figure saved to '{args.output}'")
	plt.show()


if __name__ == "__main__":
	main()
