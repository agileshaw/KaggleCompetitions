from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
import argparse
import random
import os
import csv

def buildModel(n_pixels, n_classes):
	model = Sequential()

	model.add(Dense(n_pixels, activation='relu', input_shape=(n_pixels,)))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(n_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def readSplit(dir_path, file_path):
	X = []
	y = []
	rows = []
	#rows = open(file_path).read().strip().split("\n")[1:]
	#random.shuffle(rows)mylines = []
	with open(file_path) as myfile:
		next(myfile)
		for line in myfile:
			rows.append(line)
	myfile.close()
	#rows.pop(0)
	print("Number of Entries: ", len(rows))

	(class_label, image_path) = rows[0].strip().split(",")[-2:]
	print(class_label)
	print(image_path)

	for row in rows:
		(class_label, image_path) = row.strip().split(",")[-2:]
		image_path = os.path.sep.join([dir_path, image_path])
		image = io.imread(image_path)

		image = transform.resize(image, (32, 32))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)

		X.append(image)
		y.append(int(class_label))

	X = np.array(X)
	y = np.array(y)
	return (X, y)

def saveWeightsBias(dir_path, model):
	for i, lay in enumerate(model.layers):
		weights = lay.get_weights()[0]
		biases = lay.get_weights()[1]
		print(weights.shape, biases.shape)
		np.savetxt(os.path.sep.join([dir_path, "weight"+str(i)+".csv"]), weights , fmt='%s', delimiter=',')
		np.savetxt(os.path.sep.join([dir_path, "bias"+str(i)+".csv"]), biases , fmt='%s', delimiter=',')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", required=True,
		help="dataset directory path")
	parser.add_argument("-m", "--model",
		help="path to save model")
	parser.add_argument("-w", "--weight",
		help="path to save weights and bias")
	#parser.add_argument("-p", "--plot", type=str, default="plot.png",
	#	help="path to training history plot")
	args = vars(parser.parse_args())

	train_path = os.path.sep.join([args["dataset"], "Train.csv"])
	test_path = os.path.sep.join([args["dataset"], "Test.csv"])

	print("Reading training data...")
	(X_train, y_train) = readSplit(args["dataset"], train_path)
	print("Reading test data...")
	(X_test, y_test) = readSplit(args["dataset"], test_path)

	print(X_train.shape)
	n_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
	X_train = X_train.reshape(X_train.shape[0], n_pixels).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], n_pixels).astype('float32')
	X_train = X_train / 255
	X_test = X_test / 255
	print(X_train.shape)

	n_classes = len(np.unique(y_train))
	print(n_classes)
	y_train = to_categorical(y_train, n_classes)
	y_test = to_categorical(y_test, n_classes)

	model = buildModel(n_pixels, n_classes)
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

	score = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: {} %\n Error Rate: {}".format(score[1], 1-score[1]))

	if (args["weight"]):
		saveWeightsBias(args["weight"], model)

	if (args["model"]):
		model.save(args["model"])

if __name__ == "__main__":
	main()
