from sklearn.ensemble import RandomForestClassifier
import numpy as np
import readData

from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVC

'''
This file contains functions intended to map irregular data on to a regular grid using RandomForestClassifier
'''

def mapping(xpoints,ypoints,rfc):

	'''
	Takes in a grid and a classifier that has already been trained
	uses the classifier to predict values on the grid

	the grid is in vectors (x=[-180...180] for example)
	'''
	prediction = []
	i = 1.

	for xi in xpoints:
		xpredict = []
		for yi in ypoints:
			pred = rfc.predict([xi,yi])
			xpredict.append(upLabels(round(pred)))
		prediction.append(xpredict)
		print 'Finished '+str(i/len(xpoints))+' %'
		i += 1


	return prediction

def training(rawData):

	'''
	Takes in the data from readData and sorts it to the format ML
	package wants
	then trains a classifier
	'''

	data = [(rawData['lon'][i] , rawData['lat'][i]) for i in range(len(rawData['lon']))]
	labels = [downLabels(i) for i in rawData['classif']]
	data,labels = cleanDoubles(data,labels)
	rfc = GaussianProcess(theta0=200000000000000000000000000000,corr='squared_exponential')
	#still running
	#rfc = SVC(kernel='poly')
	rfc.fit(data, labels)
	print 'Training over'
	return rfc

def upLabels(label):
	
	if label == 12.0:
		return 15.0
	if label == 13.0:
		return 20.0
	if label == 14.0:
		return 21.0
	if label == 15.0:
		return 22.0
	if label == 16.0:
		return 23.0
	return label

def downLabels(label):

	if label == 15.0:
		return 12.0
	if label == 20.0:
		return 13.0
	if label == 21.0:
		return 14.0
	if label == 22.0:
		return 15.0
	if label == 23.0:
		return 16.0
	return label

	


def cleanDoubles(data,labels):

	'''
	Takes in the data and cleans out points that appear twice
	'''

	cleanData = [data[0]]
	cleanLabels = [labels[0]]

	for i in range(len(data)):

		if data[i] not in cleanData:

			cleanData.append(data[i])
			cleanLabels.append(labels[i])

	return cleanData,cleanLabels