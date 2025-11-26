#! ./.venv/bin/python3

import sys

import pandas as pd

if __name__ == "__main__":
	my_file = "houses.csv"
	predData = pd.read_csv(my_file)

	checker = 'datasets/dataset_truth.csv'
	truthData = pd.read_csv(checker)

	predHouses = predData['Hogwarts House']
	trueHouses = truthData['Hogwarts House']
	if len(predHouses) != len(trueHouses):
		print("Error: Prediction and truth files have different number of entries.")
		sys.exit(1)
	correct = sum(predHouses == trueHouses)
	total = len(trueHouses)
	accuracy = correct / total * 100
	print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct predictions)")