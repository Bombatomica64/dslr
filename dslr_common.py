"""Shared helpers: hand-written statistics, the training feature list and CSV loading."""
import math
import sys
from collections.abc import Iterable

import pandas as pd

Column = Iterable[float | None]

HOUSE_COLUMN = 'Hogwarts House'
HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
HOUSE_COLORS = {
	'Gryffindor': 'red',
	'Hufflepuff': 'gold',
	'Ravenclaw': 'blue',
	'Slytherin': 'green',
}

# Features used for training, chosen from the visualisations (see README):
# - Arithmancy, Care of Magical Creatures: homogeneous across houses (histogram) -> no signal
# - Defense Against the Dark Arts: exactly -Astronomy/100 (scatter_plot) -> redundant, keep Astronomy
# - Potions: weakest separation of the rest (histogram spread 0.89) and no held-out gain
FEATURES = [
	'Astronomy',
	'Herbology',
	'Divination',
	'Muggle Studies',
	'Ancient Runes',
	'History of Magic',
	'Transfiguration',
	'Charms',
	'Flying',
]
DROPPED_FEATURES = {
	'Arithmancy': 'homogeneous across houses',
	'Care of Magical Creatures': 'homogeneous across houses',
	'Defense Against the Dark Arts': 'duplicate of Astronomy (r = -1)',
	'Potions': 'weakest separation left, no held-out accuracy gain',
}


def is_missing(value: float | None) -> bool:
	return value is None or (isinstance(value, float) and math.isnan(value))


def values(series: Column) -> list[float]:
	return [float(v) for v in series if not is_missing(v)]


def count(series: Column) -> int:
	n = 0
	for _ in values(series):
		n += 1
	return n


def mean(series: Column) -> float:
	total = 0.0
	n = 0
	for v in values(series):
		total += v
		n += 1
	return total / n if n else float('nan')


def std(series: Column, mean_value: float | None = None) -> float:
	"""Sample standard deviation (ddof=1), like pandas.describe."""
	if mean_value is None:
		mean_value = mean(series)
	total = 0.0
	n = 0
	for v in values(series):
		total += (v - mean_value) ** 2
		n += 1
	return (total / (n - 1)) ** 0.5 if n > 1 else float('nan')


def min_value(series: Column) -> float:
	result = float('nan')
	for v in values(series):
		if is_missing(result) or v < result:
			result = v
	return result


def max_value(series: Column) -> float:
	result = float('nan')
	for v in values(series):
		if is_missing(result) or v > result:
			result = v
	return result


def quantile(series: Column, q: float) -> float:
	"""Linear interpolation between closest ranks (pandas default)."""
	ordered = sorted(values(series))
	n = len(ordered)
	if n == 0:
		return float('nan')
	pos = (n - 1) * q
	lower = int(pos)
	if lower + 1 >= n:
		return ordered[lower]
	weight = pos - lower
	return ordered[lower] * (1 - weight) + ordered[lower + 1] * weight


def variance(series: Column, mean_value: float | None = None) -> float:
	"""Sample variance (ddof=1): the square of std."""
	return std(series, mean_value) ** 2


def central_moment(series: Column, power: int, mean_value: float) -> float:
	"""Average of (x - mean)^power over the non-missing values."""
	total = 0.0
	n = 0
	for v in values(series):
		total += (v - mean_value) ** power
		n += 1
	return total / n if n else float('nan')


def skewness(series: Column, mean_value: float | None = None) -> float:
	"""Adjusted Fisher-Pearson skewness G1 (needs n >= 3), like pandas.skew."""
	if mean_value is None:
		mean_value = mean(series)
	n = count(series)
	m2 = central_moment(series, 2, mean_value)
	if n < 3 or not m2:
		return float('nan')
	g1 = central_moment(series, 3, mean_value) / m2 ** 1.5
	return g1 * (n * (n - 1)) ** 0.5 / (n - 2)


def kurtosis(series: Column, mean_value: float | None = None) -> float:
	"""Adjusted excess kurtosis G2 (needs n >= 4), like pandas.kurt."""
	if mean_value is None:
		mean_value = mean(series)
	n = count(series)
	m2 = central_moment(series, 2, mean_value)
	if n < 4 or not m2:
		return float('nan')
	g2 = central_moment(series, 4, mean_value) / m2 ** 2 - 3
	return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)


def numeric_courses(data: pd.DataFrame) -> list[str]:
	return [c for c in data.columns
		if c != 'Index' and pd.api.types.is_numeric_dtype(data[c]) and count(data[c]) > 0]


def load_csv(filename: str) -> pd.DataFrame:
	"""Read a CSV or exit with a readable error."""
	try:
		return pd.read_csv(filename)
	except FileNotFoundError:
		sys.exit(f"Error: file '{filename}' not found.")
	except IsADirectoryError:
		sys.exit(f"Error: '{filename}' is a directory.")
	except PermissionError:
		sys.exit(f"Error: permission denied reading '{filename}'.")
	except pd.errors.EmptyDataError:
		sys.exit(f"Error: file '{filename}' is empty.")
	except (pd.errors.ParserError, UnicodeDecodeError):
		sys.exit(f"Error: unable to parse '{filename}' as CSV.")


def require_house_column(data: pd.DataFrame, filename: str) -> None:
	if HOUSE_COLUMN not in data.columns:
		sys.exit(f"Error: '{HOUSE_COLUMN}' column not found in '{filename}'.")
