import numpy as np
import readData
import mapping
import writeResults
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import RandomForestClassifier
'''
Lesa gogn
bara taka punkta innan [x0,x1]
varpa a kort
fylla inn restina med einhverju odru
bua til kort
'''

def blocks(dx,filename,theta,withData = False,proj = 'M',xBlockRange = [-180,180],yBlockRange = [-90,90]):

	xpoints = np.arange(-180,180,dx)
	ypoints = np.arange(-90+dx,90,dx)
	rawData = readData.readCols(filename)
	rfc = blockTraining(rawData,theta,xBlockRange,yBlockRange)
	prediction = blockPredict(rfc,xpoints,ypoints,xBlockRange,yBlockRange)
	writeResults.writePredictions(xpoints,ypoints,prediction, header = '> Predictions made with training.py',newFilename = 'seabed_lithology_blocks_regular_grid.txt')
	writeResults.makeMap(dx, withData, proj)
	writeResults.gmtMap()



	
def blockTraining(rawData,theta,xBlockRange,yBlockRange):

	'''
	Takes in the data from readData and sorts it to the format ML
	package wants
	then trains a classifier
	'''

	data = []
	labels = []
	x0 = xBlockRange[0]
	x1 = xBlockRange[1]
	y0 = yBlockRange[0]
	y1 = yBlockRange[1]
	for i in range(len(rawData['lon'])):
		xi = rawData['lon'][i]
		yi = rawData['lat'][i]
		if x0 < xi < x1 and y0 < yi < y1:
			data.append([xi,yi])
			labels.append(mapping.downLabels(rawData['classif'][i]))
	data,labels = mapping.cleanDoubles(data,labels)
	rfc = RandomForestClassifier()
	rfc.fit(data, labels)
	
	return rfc


def blockPredict(rfc,xpoints,ypoints,xBlockRange,yBlockRange):

	x0 = xBlockRange[0]
	x1 = xBlockRange[1]
	y0 = yBlockRange[0]
	y1 = yBlockRange[1]
	i = 1.
	prediction =[]
	for xi in xpoints:
		xpredict = []
		for yi in ypoints:
			if x0 < xi < x1 and y0 < yi < y1:
				pred = rfc.predict([xi,yi])
				xpredict.append(mapping.upLabels(round(pred)))
			else:
				xpredict.append(1.0)
		prediction.append(xpredict)
		print 'Finished '+str(i/len(xpoints))+' %'
		i += 1

	return prediction