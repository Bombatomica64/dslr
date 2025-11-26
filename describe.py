#! ./.venv/bin/python3
import sys
import pandas as pd

def count(series: pd.Series) -> int:
	ret = 0
	for value in series:
		if pd.notnull(value):
			ret += 1
	return ret

def mean(series: pd.Series) -> float:
	ret = 0.0
	n = 0
	for value in series:
		if pd.notnull(value):
			ret += value
			n += 1
	if n == 0:
		return float('nan')
	return ret / n

def std(series: pd.Series, mean_value: float) -> float:
	ret = 0.0
	n = 0
	for value in series:
		if pd.notnull(value):
			ret += (value - mean_value) ** 2
			n += 1
	if n <= 1:
		return float('nan')
	return (ret / (n - 1)) ** 0.5

def min_value(series: pd.Series) -> float:
	min_val = float('inf')
	for value in series:
		if pd.notnull(value) and value < min_val:
			min_val = value
	if min_val == float('inf'):
		return float('nan')
	return min_val

def max_value(series: pd.Series) -> float:
	max_val = float('-inf')
	for value in series:
		if pd.notnull(value) and value > max_val:
			max_val = value
	if max_val == float('-inf'):
		return float('nan')
	return max_val

def quantile(series: pd.Series, q: float) -> float:
	sorted_values = sorted([value for value in series if pd.notnull(value)])
	n = len(sorted_values)
	if n == 0:
		return float('nan')
	index = (n - 1) * q
	lower = int(index)
	upper = lower + 1
	if upper >= n:
		return sorted_values[lower]
	weight = index - lower
	return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

# count, mean, std, min, 25%, 50%, 75%, max
def describe(data: pd.DataFrame):
	args = data.columns.tolist()
	stats = list()
	for arg in args:
		if pd.api.types.is_numeric_dtype(data[arg]) and data[arg].count() > 0 and not arg == 'Index':
			print(f"Description for column: {arg}")
			column_stats = [
				count(data[arg]),
				mean(data[arg]),
				std(data[arg], mean(data[arg])),
				min_value(data[arg]),
				quantile(data[arg], 0.25),
				quantile(data[arg], 0.5),
				quantile(data[arg], 0.75),
				max_value(data[arg])
			]
			stats.append(column_stats)
			print(f"Count: {column_stats[0]}")
			print(f"Mean: {column_stats[1]}")
			print(f"Std: {column_stats[2]}")
			print(f"Min: {column_stats[3]}")
			print(f"25%: {column_stats[4]}")
			print(f"50%: {column_stats[5]}")
			print(f"75%: {column_stats[6]}")
			print(f"Max: {column_stats[7]}")
			print()  # Empty line between columns
if __name__ == "__main__":
	# sys.argv[0] is the script name
	# sys.argv[1] is the first argument
	if len(sys.argv) < 2:
		print("Usage: ./describe.py <filename>")
		sys.exit(1)
		
	filename = sys.argv[1]
	
	try:
		print(f"Processing file: {filename}")
		file = pd.read_csv(filename)
		describe(file)
	except FileNotFoundError:
		print(f"Error: File '{filename}' not found.")
		sys.exit(1)
	except pd.errors.EmptyDataError:
		print(f"Error: File '{filename}' is empty.")
		sys.exit(1)
	except pd.errors.ParserError:
		print(f"Error: Unable to parse '{filename}'. Make sure it's a valid CSV file.")
		sys.exit(1)
	except Exception as e:
		print(f"Error: An unexpected error occurred: {e}")
		sys.exit(1)

	
