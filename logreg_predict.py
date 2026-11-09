#!/usr/bin/env python3
"""logreg_predict.py dataset_test.csv weights.pkl - writes houses.csv."""
import pickle
import sys

import numpy as np
import pandas as pd

from dslr_common import load_csv
from logreg_train import prepare, sigmoid

OUTPUT = 'houses.csv'


def load_weights(filename: str) -> dict:
	try:
		with open(filename, 'rb') as f:
			model = pickle.load(f)
	except FileNotFoundError:
		sys.exit(f"Error: weights file '{filename}' not found. Run ./logreg_train.py first.")
	except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as e:
		sys.exit(f"Error: cannot load weights from '{filename}': {e}")
	if not isinstance(model, dict) or not {'houses', 'features', 'means', 'stds', 'weights'} <= model.keys():
		sys.exit(f"Error: '{filename}' is not a weights file produced by logreg_train.py.")
	return model


def predict(data: pd.DataFrame, model: dict) -> np.ndarray:
	X = prepare(data, model['means'], model['stds'])
	houses = model['houses']
	probs = np.column_stack([sigmoid(X @ model['weights'][h]) for h in houses])
	return np.array(houses)[np.argmax(probs, axis=1)]


def main() -> None:
	if len(sys.argv) != 3:
		sys.exit("Usage: ./logreg_predict.py <dataset_test.csv> <weights.pkl>")
	data = load_csv(sys.argv[1])
	model = load_weights(sys.argv[2])
	missing = [f for f in model['features'] if f not in data.columns]
	if missing:
		sys.exit(f"Error: missing feature columns: {missing}")
	predictions = predict(data, model)
	index = data['Index'] if 'Index' in data.columns else range(len(data))
	pd.DataFrame({'Index': index, 'Hogwarts House': predictions}).to_csv(OUTPUT, index=False)
	print(f"{len(predictions)} predictions written to '{OUTPUT}'")


if __name__ == "__main__":
	main()
