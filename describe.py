#!/usr/bin/env python3
"""describe.py dataset.csv [--basic] - pandas-like describe() built from hand-written statistics.

Prints the eight rows of the subject, then (bonus) Var, Range, IQR, Skew and Kurt.
--basic prints only the eight required rows.
"""
import argparse
import shutil
import sys
from collections.abc import Iterable
from typing import Any

import pandas as pd

from dslr_common import (
	count,
	kurtosis,
	load_csv,
	max_value,
	mean,
	min_value,
	numeric_courses,
	quantile,
	skewness,
	std,
	variance,
)

BASIC_ROWS = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
EXTRA_ROWS = ['Var', 'Range', 'IQR', 'Skew', 'Kurt']
COL_WIDTH = 14
LABEL_WIDTH = 6


def column_stats(series: Iterable[Any], extra: bool = True) -> list[float]:
	m = mean(series)
	low, high = min_value(series), max_value(series)
	q1, q3 = quantile(series, 0.25), quantile(series, 0.75)
	stats = [count(series), m, std(series, m), low, q1, quantile(series, 0.5), q3, high]
	if extra:
		stats += [variance(series, m), high - low, q3 - q1, skewness(series, m), kurtosis(series, m)]
	return stats


def short_name(name: str) -> str:
	width = COL_WIDTH - 1
	return name if len(name) <= width else name[:width - 1] + '.'


def describe(data: pd.DataFrame, extra: bool = True) -> None:
	features = numeric_courses(data)
	if not features:
		sys.exit("Error: no numeric feature found.")
	rows = BASIC_ROWS + EXTRA_ROWS if extra else BASIC_ROWS
	stats = {f: column_stats(data[f], extra) for f in features}
	term_width = shutil.get_terminal_size((160, 24)).columns
	per_block = max(1, (term_width - LABEL_WIDTH) // COL_WIDTH)
	for start in range(0, len(features), per_block):
		block = features[start:start + per_block]
		if start:
			print()
		print(' ' * LABEL_WIDTH + ''.join(f"{short_name(f):>{COL_WIDTH}}" for f in block))
		for r, row in enumerate(rows):
			cells = ''.join(f"{stats[f][r]:>{COL_WIDTH}.6f}" for f in block)
			print(f"{row:<{LABEL_WIDTH}}{cells}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Describe every numeric feature of a dataset.')
	parser.add_argument('dataset', help='CSV file (e.g. datasets/dataset_train.csv)')
	parser.add_argument('--basic', action='store_true', help='only the 8 rows of the subject')
	args = parser.parse_args()
	describe(load_csv(args.dataset), extra=not args.basic)
