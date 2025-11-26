#! ./.venv/bin/python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Which Hogwarts course has a homogeneous score distribution between all four houses?
class HistogramNavigator:
	def __init__(self, data, courses, houses):
		self.data = data
		self.courses = courses
		self.houses = houses
		self.current_idx = 0
		
		# Create figure and axes
		self.fig, self.ax = plt.subplots(figsize=(12, 6))
		plt.subplots_adjust(bottom=0.2)
		
		# Create buttons
		ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
		ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
		self.btn_prev = Button(ax_prev, 'Previous')
		self.btn_next = Button(ax_next, 'Next')
		
		self.btn_prev.on_clicked(self.prev_course)
		self.btn_next.on_clicked(self.next_course)
		
		# Plot first histogram
		self.plot_histogram()
		plt.show()
	
	def plot_histogram(self):
		self.ax.clear()
		course = self.courses[self.current_idx]
		
		for house in self.houses:
			house_data = self.data[self.data['Hogwarts House'] == house][course].dropna()
			self.ax.hist(house_data, alpha=0.5, label=house, bins=20)
		
		self.ax.set_title(f'{course} ({self.current_idx + 1}/{len(self.courses)})')
		self.ax.set_xlabel('Score')
		self.ax.set_ylabel('Frequency')
		self.ax.legend()
		self.fig.canvas.draw()
	
	def next_course(self, event):
		self.current_idx = (self.current_idx + 1) % len(self.courses)
		self.plot_histogram()
	
	def prev_course(self, event):
		self.current_idx = (self.current_idx - 1) % len(self.courses)
		self.plot_histogram()


if __name__ == "__main__":
	# sys.argv[0] is the script name
	# sys.argv[1] is the first argument
	if len(sys.argv) < 2:
		print("Usage: ./histogram.py <filename>")
		sys.exit(1)
		
	filename = sys.argv[1]
	
	try:
		print(f"Processing file: {filename}")
		file = pd.read_csv(filename)
		
		# Get all numeric columns (courses)
		numeric_cols = file.select_dtypes(include=['float64', 'int64']).columns
		courses = [col for col in numeric_cols if col != 'Index']
		
		if len(courses) == 0:
			print("Error: No numeric columns found in the dataset.")
			sys.exit(1)
		
		# Get houses (remove NaN)
		if 'Hogwarts House' not in file.columns:
			print("Error: 'Hogwarts House' column not found in the dataset.")
			sys.exit(1)
		
		houses = file['Hogwarts House'].unique()
		houses = [h for h in houses if pd.notna(h)]
		
		if len(houses) == 0:
			print("Error: No house data found in the dataset.")
			sys.exit(1)
		
		# Create interactive histogram navigator
		navigator = HistogramNavigator(file, courses, houses)
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
