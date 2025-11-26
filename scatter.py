#! ./.venv/bin/python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


# What are the two features that are similar ?
class ScatterNavigator:
	def __init__(self, data, courses, houses):
		self.data = data
		self.courses = courses
		self.houses = houses
		self.current_idx = 0
		
		# Calculate all correlations and sort them
		self.correlations = []
		correlation_matrix = data[courses].corr()
		for i in range(len(courses)):
			for j in range(i+1, len(courses)):
				corr_value = correlation_matrix.iloc[i, j]
				if not np.isnan(corr_value):
					self.correlations.append({
						'feature1': courses[i],
						'feature2': courses[j],
						'correlation': corr_value,
						'abs_correlation': abs(corr_value)
					})
		
		# Sort by absolute correlation (highest first)
		self.correlations = sorted(self.correlations, key=lambda x: x['abs_correlation'], reverse=True)
		
		# Create figure and axes
		self.fig, self.ax = plt.subplots(figsize=(12, 8))
		plt.subplots_adjust(bottom=0.2)
		
		# Create buttons
		ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
		ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
		self.btn_prev = Button(ax_prev, 'Previous')
		self.btn_next = Button(ax_next, 'Next')
		
		self.btn_prev.on_clicked(self.prev_pair)
		self.btn_next.on_clicked(self.next_pair)
		
		# Plot first scatter plot
		self.plot_scatter()
		plt.show()
	
	def plot_scatter(self):
		self.ax.clear()
		pair = self.correlations[self.current_idx]
		feature1 = pair['feature1']
		feature2 = pair['feature2']
		correlation = pair['correlation']
		
		# Define colors for each house
		house_colors = {
			'Gryffindor': 'red',
			'Slytherin': 'green',
			'Ravenclaw': 'blue',
			'Hufflepuff': 'yellow'
		}
		
		# Plot by house with different colors if houses exist, otherwise plot all points
		if len(self.houses) > 0 and 'Hogwarts House' in self.data.columns:
			for house in self.houses:
				house_data = self.data[self.data['Hogwarts House'] == house]
				color = house_colors.get(house, 'gray')
				self.ax.scatter(house_data[feature1], house_data[feature2], 
							   label=house, alpha=0.6, s=30, c=color, edgecolors='black', linewidth=0.5)
			self.ax.legend()
		else:
			# No house data, plot all points
			self.ax.scatter(self.data[feature1], self.data[feature2], 
						   alpha=0.6, s=30, c='blue')
		
		self.ax.set_xlabel(feature1)
		self.ax.set_ylabel(feature2)
		self.ax.set_title(f'{feature1} vs {feature2}\nCorrelation: {correlation:.4f} ({self.current_idx + 1}/{len(self.correlations)})')
		self.ax.grid(True, alpha=0.3)
		self.fig.canvas.draw()
	
	def next_pair(self, event):
		self.current_idx = (self.current_idx + 1) % len(self.correlations)
		self.plot_scatter()
	
	def prev_pair(self, event):
		self.current_idx = (self.current_idx - 1) % len(self.correlations)
		self.plot_scatter()


if __name__ == "__main__":
	# sys.argv[0] is the script name
	# sys.argv[1] is the first argument
	if len(sys.argv) < 2:
		print("Usage: ./scatter.py <filename>")
		sys.exit(1)
		
	filename = sys.argv[1]
	
	try:
		print(f"Processing file: {filename}")
		file = pd.read_csv(filename)
		
		# Get all numeric columns (courses)
		numeric_cols = file.select_dtypes(include=['float64', 'int64']).columns
		courses = [col for col in numeric_cols if col != 'Index']
		
		if len(courses) < 2:
			print("Error: Need at least 2 numeric columns to create scatter plots.")
			sys.exit(1)
		
		# Get houses (remove NaN)
		houses = file['Hogwarts House'].unique()
		houses = [h for h in houses if pd.notna(h)]
		
		# Create interactive scatter plot navigator
		navigator = ScatterNavigator(file, courses, houses)
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

