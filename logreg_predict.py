#! ./.venv/bin/python3
import sys
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, List, Any
from numpy.typing import NDArray

def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
	"""Sigmoid function for logistic regression"""
	return 1 / (1 + np.exp(-z))

def normalize_features(X: NDArray[np.float64], normalization_stats: List[Dict[str, float]]) -> NDArray[np.float64]:
	"""Normalize features using saved min-max statistics"""
	X_normalized = np.copy(X)
	
	# Skip the first column (bias term = 1)
	for i in range(1, X.shape[1]):
		stats = normalization_stats[i - 1]
		col_min = stats['min']
		col_max = stats['max']
		
		# Apply same normalization as training
		if col_max - col_min != 0:
			X_normalized[:, i] = (X[:, i] - col_min) / (col_max - col_min)
		else:
			X_normalized[:, i] = 0
	
	return X_normalized

def predict(X: NDArray[np.float64], weights_data: Dict[str, Any]) -> Tuple[Any, Any, pd.DataFrame]:
	"""Predict house for each student using trained weights
	
	Returns:
		predictions: Array of predicted house names
		max_probabilities: Array of confidence scores
		prob_df: DataFrame with probabilities for each house
	"""
	houses: List[str] = weights_data['houses']
	features: List[str] = weights_data['features']
	normalization_stats: List[Dict[str, float]] = weights_data['normalization_stats']
	all_weights: Dict[str, NDArray[np.float64]] = weights_data['weights']
	
	# Normalize features using training statistics
	X_normalized = normalize_features(X, normalization_stats)
	
	# Calculate probability for each house (One-vs-All)
	probabilities: Dict[str, NDArray[np.float64]] = {}
	for house in houses:
		theta = all_weights[house]
		# Calculate z = X @ theta
		z = np.dot(X_normalized, theta)
		# Apply sigmoid to get probability
		probabilities[house] = sigmoid(z)
	
	# Convert to DataFrame for easier manipulation
	prob_df = pd.DataFrame(probabilities)
	
	# Predict the house with highest probability
	predictions = prob_df.idxmax(axis=1).values
	max_probabilities = prob_df.max(axis=1).values
	
	return predictions, max_probabilities, prob_df

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("Usage: ./logreg_predict.py <test_data.csv>")
		sys.exit(1)

	filename = sys.argv[1]
	
	try:
		print(f"Loading test data from: {filename}")
		data = pd.read_csv(filename)
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
		print(f"Error: Unable to read file '{filename}': {e}")
		sys.exit(1)

	# Load trained weights
	try:
		print("Loading trained weights from: weights.pkl")
		with open('weights.pkl', 'rb') as f:
			weights_data = pickle.load(f)
	except FileNotFoundError:
		print("Error: Model parameters file 'weights.pkl' not found.")
		print("Please run './logreg_train.py <training_data.csv>' first to train the model.")
		sys.exit(1)
	except Exception as e:
		print(f"Error: Unable to load weights: {e}")
		sys.exit(1)
	
	# Extract features and prepare data
	features = weights_data['features']
	
	# Check if all required features exist
	missing_features = [f for f in features if f not in data.columns]
	if missing_features:
		print(f"Error: Missing required features: {missing_features}")
		sys.exit(1)
	
	# Prepare feature matrix
	X = data[features].copy()
	
	# Fill missing values with column mean (same as training)
	for col in features:
		X[col] = X[col].fillna(X[col].mean())
	
	# Convert to numpy and add bias term
	X = X.values
	X = np.column_stack([np.ones(X.shape[0]), X])
	
	# Make predictions
	print("\nMaking predictions...")
	predictions, confidences, probabilities = predict(X, weights_data)
	
	# Add predictions to original data
	data['Predicted House'] = predictions
	data['Confidence'] = confidences
	
	# Save predictions to file
	output_file = 'houses.csv'
	
	# Check if 'Index' column exists, otherwise use DataFrame index
	if 'Index' in data.columns:
		result_df = pd.DataFrame({
			'Index': data['Index'],
			'Hogwarts House': predictions
		})
	else:
		result_df = pd.DataFrame({
			'Index': range(len(predictions)),
			'Hogwarts House': predictions
		})
	
	result_df.to_csv(output_file, index=False)
	
	print(f"\n✓ Predictions saved to '{output_file}'")
	print(f"✓ Predicted {len(predictions)} students")
	
	# Display summary statistics
	print("\nPrediction Summary:")
	for house in weights_data['houses']:
		count = np.sum(predictions == house)
		percentage = (count / len(predictions)) * 100
		print(f"  {house}: {count} students ({percentage:.1f}%)")
	
	# Display confidence statistics
	print(f"\nConfidence Statistics:")
	print(f"  Mean confidence: {np.mean(confidences):.3f}")
	print(f"  Min confidence: {np.min(confidences):.3f}")
	print(f"  Max confidence: {np.max(confidences):.3f}")
	
	# Show first few predictions with details
	print("\nFirst 5 predictions:")
	print("-" * 80)
	for i in range(min(5, len(predictions))):
		idx = data['Index'].iloc[i] if 'Index' in data.columns else i
		print(f"Student {idx}:")
		print(f"  Predicted: {predictions[i]} (confidence: {confidences[i]:.3f})")
		print(f"  Probabilities: ", end="")
		for house in weights_data['houses']:
			print(f"{house}: {probabilities.iloc[i][house]:.3f} ", end="")
		print()
		print()