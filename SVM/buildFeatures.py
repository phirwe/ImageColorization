import cv2
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from parameters import *
from scipy import fftpack

def buildFeatureSpace(img, pca, scaler):
	features = []
	for image in img:
		for x in range(int(n), int(image.H-n)):
			for y in range(int(n), int(image.W-n)):
				features.append(getFeatureVector(image.L, image.Lg1, image.Lg2, x, y))

	# each row specifies an observation
	features = np.array(features)

	features = scaler.fit_transform(features)
	reduced_features = pca.fit_transform(features)

	return reduced_features

def getFeatureVector(imgL, imgLg1, imgLg2, x, y):
	fft_features = np.array(fftFromPixel(imgL, x, y))
	dct_features = np.array(dctFromPixel(imgL, x, y))
	dst_features = np.array(dstFromPixel(imgL, x, y))
	surf_features = np.array(surfFromPixel(imgL, imgLg1, imgLg2, x, y))
	dev, mean = computeStdDevAndMean(imgL, x, y)
	extra_features = np.array([dev, mean, imgL[x, y]])
	return np.concatenate((surf_features, fft_features, extra_features, dct_features, dst_features))

def surfFromPixel(imgL, imgLg1, imgLg2, x, y):
	surf = cv2.xfeatures2d.SURF_create()
	surf.setExtended(True)

	kp = cv2.KeyPoint(x, y, surfWindow)

	_, descriptor1 = surf.compute(imgL, [kp])
	if descriptor1 is None:
		descriptor1 = np.zeros((1,128))
	_, descriptor2 = surf.compute(imgLg1, [kp])
	if descriptor2 is None:
		descriptor2 = np.zeros((1,128))
	_, descriptor3 = surf.compute(imgLg2, [kp])
	if descriptor3 is None:
		descriptor3 = np.zeros((1,128))
	return np.reshape(np.concatenate((descriptor1, descriptor2, descriptor3), axis=1), -1)

def fftFromPixel(imgL, x, y):
	imgL = imgL.astype(np.float)
	neighbors = findNeighbors(imgL, x, y).flatten()
	feature = np.abs(np.fft.fft(neighbors))
	return feature

def dctFromPixel(imgL, x, y):
	imgL = imgL.astype(np.float)
	neighbors = findNeighbors(imgL, x, y).flatten()
	feature = np.abs(fftpack.dct(neighbors))
	return feature

def dstFromPixel(imgL, x, y):
	imgL = imgL.astype(np.float)
	neighbors = findNeighbors(imgL, x, y).flatten()
	feature = np.abs(fftpack.dst(neighbors))
	return feature

def findNeighbors(img, x, y, size = surfWindow):
	n = (size-1)/2
	H, W = img.shape
	x_min = int(max(x - n, 0))
	x_max = int(min(x + n + 1, H))
	y_min = int(max(y - n, 0))
	y_max = int(min(y + n + 1, W))
	neighbors = img[x_min : x_max,y_min : y_max]
	return neighbors

def computeStdDevAndMean(image, x, y):
	square = findNeighbors(image, x, y)
	return np.std(square), np.mean(square)

def testImageFeatures(image, pca, scaler):
	# features = np.zeros((image.H-2*n, image.W-2*n, numFeatures))
	features = []

	for x in range(int(n), int(image.H-n)):
		for y in range(int(n), int(image.W-n)):
			# # find feature vector
			# feature_vector = getFeatureVector(image.L, image.Lg1, image.Lg2, x, y)
			# # normalize feature vector
			# feature_vector = scaler.transform(feature_vector.astype(np.float))
			# # perform PCA
			# feature_vector = pca.transform(feature_vector)
			features.append(getFeatureVector(image.L, image.Lg1, image.Lg2, x, y))

	# each row specifies an observation
	features = np.array(features)

	features = scaler.transform(features)
	reduced_features = pca.transform(features)

	image.features = reduced_features
