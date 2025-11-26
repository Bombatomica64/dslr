#! ./.venv/bin/python3
import sys
import pandas as pd

import numpy as np

class LogisticRegressionMath:
    
    def __init__(self):
        pass

    # Sigmoid Function (hypothesis function)
    # g(z) = 1 / (1 + e^-z) 
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # The Prediction (Hypothesis h_theta)
    # h_theta(x) = g(theta^T * x) 
    def predict_proba(self, X, theta):
        # matrix multiplication (theta^T * x)
        z = np.dot(X, theta)
        return self.sigmoid(z)

    # Cost Function (Log Loss)
    # J(theta) = -1/m * sum(...) 
    def cost_function(self, X, y, theta):
        m = len(y)
        h = self.predict_proba(X, theta)
        epsilon = 1e-15
        term1 = y * np.log(h + epsilon)
        term2 = (1 - y) * np.log(1 - h + epsilon)
        cost = (-1 / m) * np.sum(term1 + term2)
        return cost

    # Batch Gradient Descent (uses entire dataset)
    # Slower but stable - good for small datasets
    def gradient_descent_batch(self, X, y, theta, learning_rate, iterations):
        m = len(y)
        cost_history = []

        for i in range(iterations):
            # 1. Calculate hypothesis (prediction) using ALL training examples
            prediction = self.predict_proba(X, theta)
            
            # 2. Calculate the error (h_theta(x) - y)
            error = prediction - y
            
            # 3. Calculate Gradient using ALL examples
            # The PDF shows sum((h - y) * x_j)
            # In vector form, this is: (X.T dot error) / m
            gradient = (1 / m) * np.dot(X.T, error)
            
            # 4. Update Weights (theta) once per iteration
            theta = theta - (learning_rate * gradient)
            
            # Track cost to ensure it's decreasing
            if i % 100 == 0:
                cost_history.append(self.cost_function(X, y, theta))
                
        return theta, cost_history

    # Stochastic Gradient Descent (uses ONE example at a time)
    # Faster, more noisy updates - good for large datasets
    # Updates weights after EACH training example
    def gradient_descent_stochastic(self, X, y, theta, learning_rate, epochs):
        m = len(y)
        cost_history = []
        
        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process ONE example at a time
            for i in range(m):
                # Get single training example
                xi = X_shuffled[i:i+1]  # Keep 2D shape
                yi = y_shuffled[i:i+1]
                
                # 1. Calculate hypothesis for this ONE example
                prediction = self.predict_proba(xi, theta)
                
                # 2. Calculate error for this ONE example
                error = prediction - yi
                
                # 3. Calculate gradient using only this ONE example
                gradient = np.dot(xi.T, error).flatten()
                
                # 4. Update weights immediately (after each example)
                theta = theta - (learning_rate * gradient)
            
            # Track cost at end of each epoch (pass through all data)
            if epoch % 10 == 0:
                cost_history.append(self.cost_function(X, y, theta))
        
        return theta, cost_history

    # Mini-Batch Gradient Descent (uses small batches)
    # Best of both worlds - faster than batch, more stable than stochastic
    # Updates weights after each MINI-BATCH of examples
    def gradient_descent_mini_batch(self, X, y, theta, learning_rate, epochs, batch_size=32):
        m = len(y)
        cost_history = []
        
        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process data in MINI-BATCHES
            for i in range(0, m, batch_size):
                # Get mini-batch (e.g., 32 examples)
                end_idx = min(i + batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                batch_m = len(y_batch)
                
                # 1. Calculate hypothesis for this mini-batch
                prediction = self.predict_proba(X_batch, theta)
                
                # 2. Calculate error for this mini-batch
                error = prediction - y_batch
                
                # 3. Calculate gradient using this mini-batch
                gradient = (1 / batch_m) * np.dot(X_batch.T, error)
                
                # 4. Update weights after each mini-batch
                theta = theta - (learning_rate * gradient)
            
            # Track cost at end of each epoch
            if epoch % 10 == 0:
                cost_history.append(self.cost_function(X, y, theta))
        
        return theta, cost_history

def normalize_features(X):
	"""Normalize features using min-max scaling"""
	X_normalized = np.copy(X)
	stats = []
	
	# Skip the first column (bias term = 1)
	for i in range(1, X.shape[1]):
		col_min = np.min(X[:, i])
		col_max = np.max(X[:, i])
		
		# Avoid division by zero
		if col_max - col_min != 0:
			X_normalized[:, i] = (X[:, i] - col_min) / (col_max - col_min)
		else:
			X_normalized[:, i] = 0
		
		stats.append({'min': col_min, 'max': col_max})
	
	return X_normalized, stats

if __name__ == "__main__":
	# sys.argv[0] is the script name
	# sys.argv[1] is the first argument
	if len(sys.argv) < 3:
		print("Usage: ./logreg_train.py <filename> <training mode>")
		sys.exit(1)
		
	filename = sys.argv[1]
	gd_method = sys.argv[2]
	
	try:
		print(f"Processing file: {filename}")
		file = pd.read_csv(filename)
		
		# Check if Hogwarts House column exists
		if 'Hogwarts House' not in file.columns:
			print("Error: 'Hogwarts House' column not found.")
			sys.exit(1)
		
		# Get all numeric columns (features)
		numeric_cols = file.select_dtypes(include=['float64', 'int64']).columns
		features = [col for col in numeric_cols if col != 'Index']
		
		if len(features) == 0:
			print("Error: No numeric features found.")
			sys.exit(1)
		
		# Remove rows with missing house labels
		file = file[file['Hogwarts House'].notna()]
		
		# Get unique houses
		houses = sorted(file['Hogwarts House'].unique())
		print(f"Training for houses: {houses}")
		print(f"Using features: {features}")
		
		# Prepare features (X) - fill NaN with column mean
		X = file[features].copy()
		for col in features:
			X[col] = X[col].fillna(X[col].mean())
		
		# Convert to numpy array and add bias term (column of 1s)
		X = X.values
		X = np.column_stack([np.ones(X.shape[0]), X])
		
		# Normalize features
		X, normalization_stats = normalize_features(X)
		
		# Train one model per house (One-vs-All)
		lr = LogisticRegressionMath()
		all_weights = {}
		
		# Choose gradient descent method:
		# 'batch' - uses all data (stable, slower)
		# 'stochastic' - uses one example at a time (faster, noisy)
		# 'mini_batch' - uses small batches (best balance)
		if gd_method not in ['batch', 'stochastic', 'mini_batch']:
			print("Error: Invalid training mode. Choose 'batch', 'stochastic', or 'mini_batch'.")
			sys.exit(1)
		learning_rate = 0.01
		epochs = 100
		batch_size = 32
		
		print(f"\nUsing {gd_method} gradient descent")
		print(f"Learning rate: {learning_rate}, Epochs: {epochs}")
		if gd_method == 'mini_batch':
			print(f"Batch size: {batch_size}")
		
		for house in houses:
			print(f"\nTraining model for {house}...")
			
			# Create binary labels (1 if this house, 0 otherwise)
			y = (file['Hogwarts House'] == house).astype(int).values
			
			# Initialize weights (theta) to zeros
			theta = np.zeros(X.shape[1])
			
			# Train using selected gradient descent method
			if gd_method == 'batch':
				theta, cost_history = lr.gradient_descent_batch(X, y, theta, learning_rate, epochs)
			elif gd_method == 'stochastic':
				theta, cost_history = lr.gradient_descent_stochastic(X, y, theta, learning_rate, epochs)
			else:  # mini_batch
				theta, cost_history = lr.gradient_descent_mini_batch(X, y, theta, learning_rate, epochs, batch_size)
			
			all_weights[house] = theta
			print(f"Final cost: {cost_history[-1]:.6f}")
			print(f"Weights shape: {theta.shape}")
		
		# Save weights to a file
		weights_data = {
			'houses': houses,
			'features': features,
			'normalization_stats': normalization_stats,
			'weights': all_weights
		}
		
		import pickle
		with open('weights.pkl', 'wb') as f:
			pickle.dump(weights_data, f)
		
		print("\n✓ Training complete!")
		print("✓ Weights saved to 'weights.pkl'")
		
		# Display weight summary
		print("\nWeight Summary:")
		for house in houses:
			print(f"\n{house}:")
			print(f"  Bias: {all_weights[house][0]:.4f}")
			for i, feature in enumerate(features, 1):
				print(f"  {feature}: {all_weights[house][i]:.4f}")

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
		import traceback
		traceback.print_exc()
		sys.exit(1)
