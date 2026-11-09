#!/usr/bin/env python3
"""logreg_train.py dataset_train.csv [batch|mini_batch|stochastic] [-o weights.pkl]

One-vs-all logistic regression trained with gradient descent.
Batch gradient descent is the default (mandatory part); mini_batch and
stochastic are the bonus optimisers.
"""
import argparse
import pickle
import sys

import numpy as np

from dslr_common import FEATURES, HOUSE_COLUMN, load_csv, mean, require_house_column, std

MODES = ('batch', 'mini_batch', 'stochastic')
# (learning rate, iterations/epochs) per mode, tuned on standardised features.
HYPERPARAMS = {
	'batch': (0.5, 1000),
	'mini_batch': (0.1, 50),
	'stochastic': (0.01, 10),
}
BATCH_SIZE = 32
SEED = 42


def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))


def cost(X, y, theta) -> float:
	h = sigmoid(X @ theta)
	eps = 1e-15
	return float(-np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)) / len(y))


def gradient(X, y, theta):
	return X.T @ (sigmoid(X @ theta) - y) / len(y)


def batch_gd(X, y, lr, iterations, rng):
	theta = np.zeros(X.shape[1])
	for _ in range(iterations):
		theta -= lr * gradient(X, y, theta)
	return theta


def mini_batch_gd(X, y, lr, epochs, rng, batch_size=BATCH_SIZE):
	theta = np.zeros(X.shape[1])
	m = len(y)
	for _ in range(epochs):
		order = rng.permutation(m)
		for start in range(0, m, batch_size):
			idx = order[start:start + batch_size]
			theta -= lr * gradient(X[idx], y[idx], theta)
	return theta


def stochastic_gd(X, y, lr, epochs, rng):
	return mini_batch_gd(X, y, lr, epochs, rng, batch_size=1)


OPTIMISERS = {'batch': batch_gd, 'mini_batch': mini_batch_gd, 'stochastic': stochastic_gd}


def prepare(data, means, stds):
	"""Fill NaN with the given (training) means, standardise, add the bias column."""
	X = np.empty((len(data), len(FEATURES)))
	for j, feature in enumerate(FEATURES):
		col = data[feature].to_numpy(dtype=float)
		col = np.where(np.isnan(col), means[feature], col)
		X[:, j] = (col - means[feature]) / stds[feature]
	return np.column_stack([np.ones(len(data)), X])


def train(data, mode='batch', seed=SEED, verbose=True):
	data = data[data[HOUSE_COLUMN].notna()]
	means = {f: mean(data[f]) for f in FEATURES}
	stds = {f: std(data[f], means[f]) or 1.0 for f in FEATURES}
	X = prepare(data, means, stds)
	houses = sorted(data[HOUSE_COLUMN].unique())
	lr, iterations = HYPERPARAMS[mode]
	rng = np.random.default_rng(seed)
	weights = {}
	for house in houses:
		y = (data[HOUSE_COLUMN] == house).to_numpy(dtype=float)
		weights[house] = OPTIMISERS[mode](X, y, lr, iterations, rng)
		if verbose:
			print(f"  {house:<11} final cost {cost(X, y, weights[house]):.4f}")
	return {
		'mode': mode,
		'houses': houses,
		'features': list(FEATURES),
		'means': means,
		'stds': stds,
		'weights': weights,
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
	parser.add_argument('dataset', help='training CSV (e.g. datasets/dataset_train.csv)')
	parser.add_argument('mode', nargs='?', default='batch', choices=MODES,
		help='gradient descent variant (default: batch)')
	parser.add_argument('-o', '--output', default='weights.pkl', help='weights file (default: weights.pkl)')
	parser.add_argument('--seed', type=int, default=SEED, help='RNG seed for shuffling (default: 42)')
	args = parser.parse_args()

	data = load_csv(args.dataset)
	require_house_column(data, args.dataset)
	missing = [f for f in FEATURES if f not in data.columns]
	if missing:
		sys.exit(f"Error: missing feature columns: {missing}")
	if data[HOUSE_COLUMN].notna().sum() == 0:
		sys.exit(f"Error: no labelled rows in '{args.dataset}'.")

	lr, iterations = HYPERPARAMS[args.mode]
	print(f"Training {args.mode} gradient descent (lr={lr}, "
		f"{'iterations' if args.mode == 'batch' else 'epochs'}={iterations}) on {len(FEATURES)} features")
	model = train(data, args.mode, args.seed)
	try:
		with open(args.output, 'wb') as f:
			pickle.dump(model, f)
	except OSError as e:
		sys.exit(f"Error: cannot write '{args.output}': {e}")
	print(f"Weights saved to '{args.output}'")


if __name__ == "__main__":
	main()
