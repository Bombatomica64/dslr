#!/usr/bin/env python3
"""describe.py dataset.csv - pandas-like describe() built from hand-written statistics."""
import shutil
import sys

from dslr_common import (count, load_csv, max_value, mean, min_value,
	numeric_courses, quantile, std)

ROWS = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
COL_WIDTH = 14
LABEL_WIDTH = 6


def column_stats(series) -> list:
	m = mean(series)
	return [count(series), m, std(series, m), min_value(series),
		quantile(series, 0.25), quantile(series, 0.5), quantile(series, 0.75),
		max_value(series)]


def short_name(name: str) -> str:
	width = COL_WIDTH - 1
	return name if len(name) <= width else name[:width - 1] + '.'


def describe(data) -> None:
	features = numeric_courses(data)
	if not features:
		sys.exit("Error: no numeric feature found.")
	stats = {f: column_stats(data[f]) for f in features}
	term_width = shutil.get_terminal_size((160, 24)).columns
	per_block = max(1, (term_width - LABEL_WIDTH) // COL_WIDTH)
	for start in range(0, len(features), per_block):
		block = features[start:start + per_block]
		if start:
			print()
		print(' ' * LABEL_WIDTH + ''.join(f"{short_name(f):>{COL_WIDTH}}" for f in block))
		for r, row in enumerate(ROWS):
			cells = ''.join(f"{stats[f][r]:>{COL_WIDTH}.6f}" for f in block)
			print(f"{row:<{LABEL_WIDTH}}{cells}")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: ./describe.py <dataset.csv>")
	describe(load_csv(sys.argv[1]))
