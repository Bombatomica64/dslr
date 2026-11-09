#!/usr/bin/env python3
"""histogram.py dataset_train.csv [-o out.png]

Which Hogwarts course has a homogeneous score distribution between all four houses?
"""
import argparse

import matplotlib.pyplot as plt

from dslr_common import (HOUSE_COLORS, HOUSE_COLUMN, load_csv, mean, numeric_courses,
	require_house_column, std)


def heterogeneity(data, course, houses) -> float:
	"""Spread of the per-house means, in units of the overall std (0 = homogeneous)."""
	house_means = [mean(data.loc[data[HOUSE_COLUMN] == h, course]) for h in houses]
	return std(house_means) / std(data[course])


def main():
	parser = argparse.ArgumentParser(description='Histogram of every course, per house.')
	parser.add_argument('dataset')
	parser.add_argument('-o', '--output', help='save the figure instead of only showing it')
	args = parser.parse_args()

	data = load_csv(args.dataset)
	require_house_column(data, args.dataset)
	houses = sorted(data[HOUSE_COLUMN].dropna().unique())
	courses = numeric_courses(data)
	if not courses or not houses:
		raise SystemExit("Error: no course or house data found.")

	scores = {c: heterogeneity(data, c, houses) for c in courses}
	ranking = sorted(courses, key=scores.get)
	answer = ranking[0]
	print("Spread of house means / overall std (lower = more homogeneous):")
	for c in ranking:
		print(f"  {c:<30} {scores[c]:.3f}")
	print(f"Answer: '{answer}' has the most homogeneous score distribution between the four houses"
		f" (runner-up: '{ranking[1]}').")

	cols = 4
	rows = (len(courses) + cols - 1) // cols
	fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
	for ax, course in zip(axes.flat, courses):
		for house in houses:
			ax.hist(data.loc[data[HOUSE_COLUMN] == house, course].dropna(), bins=25, alpha=0.5,
				color=HOUSE_COLORS.get(house), label=house)
		ax.set_title(f"{course}  (spread {scores[course]:.2f})",
			fontsize=10, fontweight='bold' if course == answer else 'normal',
			color='darkred' if course == answer else 'black')
		ax.set_ylabel('Students')
	for ax in list(axes.flat)[len(courses):]:
		ax.axis('off')
	axes.flat[0].legend()
	fig.suptitle(f"Most homogeneous course across the four houses: {answer}", fontsize=16)
	fig.tight_layout(rect=(0, 0, 1, 0.97))
	if args.output:
		fig.savefig(args.output)
		print(f"Figure saved to '{args.output}'")
	plt.show()


if __name__ == "__main__":
	main()
