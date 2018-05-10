from sklearn import svm
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn import preprocessing
import cv2
import numpy as np
import image
from parameters import *
from buildFeatures import *



def kmeans(img,K):
	kmeans = cluster.KMeans(K)
	kmeansfeaturesall = []
	kmeansfeaturesall = img[0].kmeansfeatures
	for i in range(1, numberOfTrainingImages):
		kmeansfeaturesall = np.append(kmeansfeaturesall, img[i].kmeansfeatures, axis=0)
	
	print(kmeansfeaturesall.shape)

	labels = kmeans.fit_predict(kmeansfeaturesall)
	return kmeans.cluster_centers_, labels

def svm_train(img, featuresall, labels):
	svm_clf = svm.LinearSVC(dual = False, class_weight = 'balanced')
	# img.features = img.labels.reshape((img.pixel_num,1))
	print("featuressize", featuresall.shape)
	svm_clf.fit(featuresall,labels)
	return svm_clf

def svm_predict(img,svm_clf):
	img.labels = svm_clf.predict(img.features)
	return svm_clf.decision_function(img.features)

def lab2color(test_img,centers):
	n = int((surfWindow - 1)/2)
	test_lab = np.zeros((test_img.H,test_img.W,3))
	H = test_img.H
	W = test_img.W
	for x in range(0,H - 2*n):
		for y in range(0,W - 2*n):
			test_lab[x+n,y+n,1:3] = centers[test_img.labels[(W-2*n)*x+y]]
	test_lab[:,:,0] = test_img.L
	return cv2.cvtColor(test_lab.astype(np.uint8),cv2.COLOR_Lab2BGR)

def lab2color2D(test_img,centers):
	H = test_img.H
	W = test_img.W
	test_lab = np.zeros((H,W,3))
	for x in range(0,H):
		for y in range(0,W):
			test_lab[x,y,1:3] = centers[test_img.labels[x*W+y]]
			# test_lab[x,y,1:3] = centers[test_img.labels[x,y]] 
	test_lab[:,:,0] = test_img.L
	test_lab = test_lab.astype(np.uint8)
	return cv2.cvtColor(test_lab,cv2.COLOR_Lab2BGR)

if __name__ == '__main__':
	#### read image
	numberOfTrainingImages = 2
	myimage = "image"
	train_img =[]
	for i in range(0, numberOfTrainingImages):
		print(i)
		imagepath = './results/multiple/train' + str(i) + '.jpg'
		img = myimage + str(i)
		img = image.Image(imagepath)
		train_img.append(img)
	# cv2.imshow('training image',train_img.bgr)
	# cv2.waitKey(0)

	#### color discretization
	print ('Start Computing K-Means of K =',K)
	centers, labels = kmeans(train_img,K)
	print ('End computing K-Means of K =',K)

	#### build feature space
	print ('Start building training features')
	pca_train = PCA(n_components = numFeatures ,whiten = True)
	min_max_scaler = preprocessing.MinMaxScaler()
	featuresall = buildFeatureSpace(train_img,pca_train,min_max_scaler)
	print ('End building Training features')

	#### perform SVM training
	print ('Start Training image')
	svm_clf = svm_train(train_img, featuresall, labels)
	print ('End Training image')
	del train_img
	#### read test image
	test_img = image.Image('./results/multiple/test.jpg')
	# cv2.imshow('testing image',test_img.bgr)
	# cv2.waitKey(0)

	#### build test image features
	print ('Start bulding Test features')
	testImageFeatures(test_img,pca_train,min_max_scaler)

	#### SVM predict and get score
	print ('Start Predict labels for test image')
	score = -1 * svm_predict(test_img,svm_clf)
	print ('End Predict labels for test image')
	svm_img = lab2color(test_img,centers)
	cv2.imwrite('./results/multiple/horse_svm_output_multiple_linear.jpg',svm_img)
	#### perform graph cut
	
	cv2.destroyAllWindows()
