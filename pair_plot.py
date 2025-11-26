#! ./.venv/bin/python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: ./pair_plot.py <filename>")
		sys.exit(1)

	filename = sys.argv[1]
	
	try:
		data = pd.read_csv(filename)

		# Get all numeric columns (courses)
		numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
		courses = [col for col in numeric_cols if col != 'Index']
		
		# Filter out columns with too many NaN values
		courses = [col for col in courses if data[col].count() > 0]
		
		if len(courses) < 2:
			print("Error: Need at least 2 numeric columns to create pair plot.")
			sys.exit(1)
		
		# Get houses (remove NaN)
		houses = []
		if 'Hogwarts House' in data.columns:
			houses = data['Hogwarts House'].unique()
			houses = [h for h in houses if pd.notna(h)]
		
		# Define colors for houses
		house_colors = {
			'Gryffindor': 'red',
			'Slytherin': 'green',
			'Ravenclaw': 'blue',
			'Hufflepuff': 'gold'
		}
		extra_colors = ['purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'lime', 'navy', 'teal', 'coral']
		
		# Create the pair plot grid
		n = len(courses)
		fig, axes = plt.subplots(n, n, figsize=(20, 20))
		fig.suptitle('Pair Plot - Histograms on Diagonal, Scatter Plots for Correlations', fontsize=16, y=0.995)
		
		for i in range(n):
			for j in range(n):
				ax = axes[i, j] if n > 1 else axes
				
				if i == j:
					# Diagonal: histogram for each house
					if len(houses) > 0:
						color_idx = 0
						for house in houses:
							house_data = data[data['Hogwarts House'] == house][courses[i]].dropna()
							if house in house_colors:
								color = house_colors[house]
							else:
								color = extra_colors[color_idx % len(extra_colors)]
								color_idx += 1
							ax.hist(house_data, alpha=0.5, bins=20, color=color, label=house if j == n-1 else "")
					else:
						ax.hist(data[courses[i]].dropna(), alpha=0.6, bins=20, color='blue')
					ax.set_ylabel('Frequency', fontsize=8)
					if i == 0:
						ax.set_title(courses[i], fontsize=9, fontweight='bold')
				else:
					# Off-diagonal: scatter plot
					if len(houses) > 0:
						color_idx = 0
						for house in houses:
							house_data = data[data['Hogwarts House'] == house]
							if house in house_colors:
								color = house_colors[house]
							else:
								color = extra_colors[color_idx % len(extra_colors)]
								color_idx += 1
							ax.scatter(house_data[courses[j]], house_data[courses[i]], 
									 alpha=0.4, s=5, c=color, edgecolors='none')
					else:
						ax.scatter(data[courses[j]], data[courses[i]], 
								 alpha=0.4, s=5, c='blue', edgecolors='none')
				
				# Labels
				if j == 0:
					ax.set_ylabel(courses[i], fontsize=8)
				else:
					ax.set_ylabel('')
				
				if i == n-1:
					ax.set_xlabel(courses[j], fontsize=8)
				else:
					ax.set_xlabel('')
				
				# Smaller tick labels
				ax.tick_params(labelsize=6)
		
		# Add legend for houses (if available)
		if len(houses) > 0:
			handles = []
			labels = []
			color_idx = 0
			for house in houses:
				if house in house_colors:
					color = house_colors[house]
				else:
					color = extra_colors[color_idx % len(extra_colors)]
					color_idx += 1
				handles.append(mlines.Line2D([0], [0], marker='o', color='w', 
										  markerfacecolor=color, markersize=8, alpha=0.6))
				labels.append(house)
			fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
		
		plt.tight_layout()
		plt.show()
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