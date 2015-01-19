import readData
import mapping
import writeResults
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.gaussian_process import GaussianProcess


'''
This file is for running the functions for mapping lithology data on to a regular grid in the order
readData 		for reading data
mapping 		for mapping data
writeResults 	for writing data
'''

def testing():

	print 'Tests deleted and need to be rewritten'
	# Test readCols
	# Test mapping
	# Test training
	# Test makeMap
	# Test gmtMap
	# Test writePredictions


	#Tests have passed
	print 'All tests passed'





def running(dx,withData = True,compare = True,proj = 'M',filename = 'fullData.txt'):

	'''
	dx gives the resolution, dx=1 for 1x1 grid
	withData is a boolean, withData=True plots the data points on top the map
	projection has the same options as gmt, Mercator projection is the default
	filename is the name of the file that stores the data
	'''

	'''
	Reading data
	'''
	theta0 = [600000]
	for theta in theta0:
		xpoints = np.arange(-180,180,dx)
		ypoints = np.arange(-90+dx,90,dx)
		if filename.endswith('txt'):
			rawData = readData.readTxt(filename)
		elif filename.endswith('xlsx'):
			rawData = readData.readCols(filename)
		else:
			print "I don't think I can handle this format"
			return
		'''
		Classifier being trained
		'''
		rfc = mapping.training(rawData)
		'''
		Map being made with the classifier
		'''
		data = [(rawData['lon'][i] , rawData['lat'][i]) for i in range(len(rawData['lon']))]
		labels = [mapping.downLabels(i) for i in rawData['classif']]
		data,labels = mapping.cleanDoubles(data,labels)
		scores = cross_val_score(rfc,data,labels,cv=5)
		print("Accuracy: %0.2f (+/- %0.2f)"
    	  % (scores.mean(), scores.std()*2))
		prediction = mapping.mapping(xpoints,ypoints,rfc)
		'''
		Map being written to a file
		'''
		writeResults.writePredictions(xpoints,ypoints,prediction, header = '> Predictions made with training.py',newFilename = 'seabed_lithology_regular_grid.txt')
		'''
		GMT file written
		'''
		writeResults.makeMap(dx, withData, proj)
		'''
		Postscript called to call gmt
		'''
		writeResults.gmtMap()
		print len(rawData['lon']),len(rawData['classif'])
		'''
		Statisctics being compared
		'''	
		if compare:
			compareStats(rawData,theta)

def compareStats(rawData,theta):

	data = [[rawData['lon'][i] , rawData['lat'][i]] for i in range(len(rawData['lon']))]
	labels = rawData['classif']
	
	data,labels = mapping.cleanDoubles(data,labels)

	rfc = GaussianProcess(regr='linear',theta0 = theta)
	rfc.fit(data,labels)
	scores = cross_val_score(rfc,data,labels,cv=5)
	print("Accuracy: %0.2f (+/- %0.2f)"
      % (scores.mean(), scores.std()*2))