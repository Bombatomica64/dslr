"""Shared helpers: hand-written statistics, the training feature list and CSV loading."""
import sys
import math

import pandas as pd

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
# - Potions: heavy overlap between houses in the pair plot -> weak signal
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
	'Potions': 'strong overlap between houses',
}


def is_missing(value) -> bool:
	return value is None or (isinstance(value, float) and math.isnan(value))


def values(series) -> list:
	return [float(v) for v in series if not is_missing(v)]


def count(series) -> int:
	n = 0
	for _ in values(series):
		n += 1
	return n


def mean(series) -> float:
	total = 0.0
	n = 0
	for v in values(series):
		total += v
		n += 1
	return total / n if n else float('nan')


def std(series, mean_value=None) -> float:
	"""Sample standard deviation (ddof=1), like pandas.describe."""
	if mean_value is None:
		mean_value = mean(series)
	total = 0.0
	n = 0
	for v in values(series):
		total += (v - mean_value) ** 2
		n += 1
	return (total / (n - 1)) ** 0.5 if n > 1 else float('nan')


def min_value(series) -> float:
	result = float('nan')
	for v in values(series):
		if is_missing(result) or v < result:
			result = v
	return result


def max_value(series) -> float:
	result = float('nan')
	for v in values(series):
		if is_missing(result) or v > result:
			result = v
	return result


def quantile(series, q: float) -> float:
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


def numeric_courses(data: pd.DataFrame) -> list:
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
