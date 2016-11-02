# Arturo Polanco Lozano Capstone Udacity
# to execute type 
# python arturo_polaco_capstone.py --dataset 4scenes

# import packages
from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import cross_validation
from imutils import paths
import numpy as np
import argparse
import mahotas
import cv2

def describe(image):
	# extract the mean and standard deviation from each channel of the image in the HSV color space
	(means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
	colorStats = np.concatenate([means, stds]).flatten()
	# extract Haralick texture features
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haralick = mahotas.features.haralick(gray).mean(axis=0)
	# return a concatenated feature vector of color statistics and Haralick
	# texture features
	return np.hstack([colorStats, haralick])

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to 4 scene category dataset or any other dataset")
args = vars(ap.parse_args())
# grab the set of image paths and initialize the list of labels and matrix of features
print("[Please Wait ....] Extracting Features")
imagePaths = sorted(paths.list_images(args["dataset"]))
labels = []
data = []



# loop over the images in the input directory
for imagePath in imagePaths:
	# extract the label and load the image from disk
	label = imagePath[imagePath.rfind("/") + 1:].split("_")[0]
	image = cv2.imread(imagePath)
	# extract features from the image, then update the list of lables and features
	features = describe(image)
	labels.append(label)
	data.append(features)

# construct the training and testing split by taking 75% of the data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25, random_state=42)

# initialize the model as a random forest
#
# train random forest
print("[Please Wait ....] Training ")
#model.fit(trainData, trainLabels)
# evaluate the classifier


num_folds = 10
num_instances = len(np.array(data))
seed = 7
num_trees = 100
max_features = 3
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_validation.cross_val_score(model, np.array(data), np.array(labels), cv=kfold)
print (results)
print(results.mean())



'''
# loop over a few random images
for i in np.random.randint(0, high=len(imagePaths), size=(10,)):
	# grab the image and classify it
	imagePath = imagePaths[i]
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	features = describe(image)
	prediction = model.predict(features.reshape(1, -1))[0]
	# show the prediction
	print("[PREDICTION] {}: {}".format(filename, prediction))
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("4Scene Image Classification ", image)
	cv2.waitKey(0)
'''

