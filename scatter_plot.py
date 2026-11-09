#!/usr/bin/env python3
"""scatter_plot.py dataset_train.csv [-o out.png]

What are the two features that are similar?
"""
import argparse
import itertools

import matplotlib.pyplot as plt

from dslr_common import (HOUSE_COLORS, HOUSE_COLUMN, is_missing, load_csv, mean,
	numeric_courses, require_house_column)


def pearson(xs, ys) -> float:
	pairs = [(x, y) for x, y in zip(xs, ys) if not is_missing(x) and not is_missing(y)]
	mx = mean([p[0] for p in pairs])
	my = mean([p[1] for p in pairs])
	sxy = sum((x - mx) * (y - my) for x, y in pairs)
	sxx = sum((x - mx) ** 2 for x, _ in pairs)
	syy = sum((y - my) ** 2 for _, y in pairs)
	return sxy / (sxx * syy) ** 0.5


def main():
	parser = argparse.ArgumentParser(description='Scatter plot of the two most similar features.')
	parser.add_argument('dataset')
	parser.add_argument('-o', '--output', help='save the figure instead of only showing it')
	args = parser.parse_args()

	data = load_csv(args.dataset)
	require_house_column(data, args.dataset)
	courses = numeric_courses(data)
	if len(courses) < 2:
		raise SystemExit("Error: need at least 2 numeric features.")

	corr = {(a, b): pearson(data[a].tolist(), data[b].tolist())
		for a, b in itertools.combinations(courses, 2)}
	ranking = sorted(corr, key=lambda k: -abs(corr[k]))
	(f1, f2) = ranking[0]
	print("Most correlated feature pairs (Pearson r):")
	for a, b in ranking[:5]:
		print(f"  {a} / {b}: {corr[(a, b)]:+.4f}")
	print(f"Answer: the two similar features are '{f1}' and '{f2}' (r = {corr[(f1, f2)]:+.4f}).")

	fig, ax = plt.subplots(figsize=(10, 8))
	for house in sorted(data[HOUSE_COLUMN].dropna().unique()):
		rows = data[data[HOUSE_COLUMN] == house]
		ax.scatter(rows[f1], rows[f2], s=12, alpha=0.6, color=HOUSE_COLORS.get(house), label=house)
	ax.set_xlabel(f1)
	ax.set_ylabel(f2)
	ax.grid(alpha=0.3)
	ax.legend()
	ax.set_title(f"Similar features: {f1} and {f2} (r = {corr[(f1, f2)]:+.4f})")
	fig.tight_layout()
	if args.output:
		fig.savefig(args.output)
		print(f"Figure saved to '{args.output}'")
	plt.show()


if __name__ == "__main__":
	main()
