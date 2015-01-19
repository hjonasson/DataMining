from mpl_toolkits.basemap import Basemap
from scipy.io import netcdf_file as netcdf
import numpy as np
import sklearn
import re
import itertools
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


'''
Look at what classifications occur most
If I get a bathymetry map, look at it in relation to that

taka kort og stadsetningar
fjarlaegja lond
skoda tidnir
bua jafnvel til kort sem hafa hverja flokkun fyrir sig

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

'''

classifications = ('Gravel','Sand','Silt','Silicious clay','Calcareous ooze','Radiolarian ooze','Diatom ooze','Sponge ooze','Mixed calc./sili. ooze','Shells and coral fragments','Ash,glass,volcanics','Mud','Fine-grained calcareous sediment')
colors = ["#808284","#FFF100","#FAA918","#704B2A","#0E91CF","#0D9647","#BED753","#55938D","#8370B2","#F7BBD5","#D83A26","#C39A6B","#002EA7"]
oceans = ['Arctic','NAtl','SAtl','NPac','SPac','NInd','SInd','SOcean']
y_pos = np.arange(len(classifications))
m = Basemap(projection='merc',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,lat_ts=5,resolution='c')#res c,l,i,h,f, has big effects on efficiency
space = ' '
newLine = '\n'

def readTxt(filename):

	'''
	Takes in a filename of a three column format and gives
	the data from it in a dictionary

	seabed_lithology_v4.txt for example
	'''

	data = {'lon':[],'lat':[],'classif':[]}
	f = open(filename)
	lines = f.readlines()
	for line in lines:
		if line[0] != '>':
			splitLine = re.split(r'\t+',line.strip())[0].split()
			if splitLine[2] != 'NaN':
				data['lon'].append(float(splitLine[0]))
				data['lat'].append(float(splitLine[1]))
				data['classif'].append(float(splitLine[2]))
	return data

def readTxt1Deg(filename):

	'''
	Takes in a filename of a three column format and gives
	the data from it in a dictionary

	seabed_lithology_v4.txt for example
	'''

	data = {'lon':[],'lat':[],'classif':[]}
	f = open(filename)
	lines = f.readlines()
	for line in lines:
		if line[0] != '>':
			splitLine = re.split(r'\t+',line.strip())[0].split()
			if not float(splitLine[0]) % 1 and not float(splitLine[1]) % 1:
				data['lon'].append(float(splitLine[0]))
				data['lat'].append(float(splitLine[1]))
				data['classif'].append(float(splitLine[2]))
	return data

def readTxtTabs(filename):

	'''
	Takes in a filename of a three column format and gives
	the data from it in a dictionary

	seabed_lithology_v4.txt for example
	'''

	data = {'lon':[],'lat':[],'classif':[]}
	f = open(filename)
	lines = f.readlines()
	for line in lines:
		if line[0] != '>':
			splitLine = re.split(r'\t+',line.strip())
			if splitLine[2] != 'NaN':
				data['lon'].append(float(splitLine[0]))
				data['lat'].append(float(splitLine[1]))
				data['classif'].append(float(splitLine[2]))
	return data

def updateData(filename = 'seabed_lithology_finegrid.txt'):

	f = open(filename)
	g = open(filename[:-4]+'upd.txt','w')
	lines = f.readlines()
	for line in lines:
		splitLine = re.split(r'\t+',line.strip())[0].split()		
		g.write(splitLine[0]+' '+splitLine[1]+' '+newClassif(splitLine[2])+'\n')
	g.close()

def newClassif(a):
	if a == 19.0:
		return  1.0
	if a == 18.0:
		return  2.0
	if a == 8.0:
		return  9.0
	if a == 11.0:
		return  10.0
	if a == 20.0:
		return  11.0
	if a == 9.0:
		return  12.0
	if a == 22.0:
		return  8.0
	if a == 10.0:
		return  13.0
	if a == 21.0:
		return 13.0
	if a == 23.0:
		return 13.0
	else:
		return a





def dataPoints(filename='seabed_lithology_points.txt'):

	rawData = readTxt(filename)
	rates = ratesOfPredictions([rawData['classif']])
	print len(rates)
	for k,v in rates.iteritems(): print k
	rate = [i[1] for i in rates.iteritems()]
	plt.barh(y_pos, rate, align='center', alpha=0.4,color=colors)
	plt.yticks(y_pos, classifications)
	plt.xlabel('Rate of classification')
	plt.title('How often does each classification appear?')

	plt.show()


def cleanContinents(m,xpoints,ypoints,predictions):

	cleanPredictions = []
	seaMap = []
	for i in range(len(xpoints)):
		xi = xpoints[i]
		pred = []
		y = []
		for j in range(len(ypoints)):
			yj = ypoints[j]
			if not Basemap.is_land(m,m(xi,yj)[0],m(xi,yj)[1]):
				pred.append(predictions[i][j])
				y.append(yj)
		cleanPredictions.append(pred)
		seaMap.append(y)

	return cleanPredictions,seaMap

def ratesOfPredictions(cleanPredictions):

	rates = {}
	for xi in cleanPredictions:
		for yj in xi:
			if yj not in rates:
				rates[yj] = 1
			else:
				rates[yj] += 1

	return rates

'''
Analyze each ocean
'''



def oceanBoundaries(x,y):

	if y >= 65.:
		return 'Arctic'
	if 0. >= x >= -60.:
		if 0. >= y >= 65.5:
			return 'NAtl'
	if -70. >= x >= 20.:
		if -40 >= y >= 0.:
			return 'SAtl'
	if 0. >= y >= 59.5:
		if x > 120.:
			return 'NPac'
		if -60. >= x:
			return 'NPac'
	if -40. >= y >= 0.:
		if x > 120.:
			return 'SPac'
		if -70. >= x:
			return 'SPac'
	if 20. >= x >= 120.:
		if 24.5 >= y >= 0.:
			return 'NInd'
		if 0. >= y >= -40.:
			return 'SInd'
	if -40. >= y:
		return 'SOcean'
	else:
		return 'other'

def bucketOceans(m,rawData):

	oceanDict = {ocean:{} for ocean in oceans}

	for i in range(len(rawData['lat'])):
		x = rawData['lon'][i]
		y = rawData['lat'][i]
		classif = rawData['classif'][i]
		ocean = oceanBoundaries(x,y)
		if not Basemap.is_land(m,m(x,y)[0],m(x,y)[1]):
			if ocean in oceans:
				if classif not in oceanDict[ocean]:
					oceanDict[ocean][classif] = 1
				else:
					oceanDict[ocean][classif] += 1
		print float(i)/len(rawData['lat'])
	print 'Bucketing done with a bucket of len '+str(len(oceanDict))
	return oceanDict

def plotOceans(dataFile,predFile):

	m = Basemap(projection='merc',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,lat_ts=5,resolution='c')#res c,l,i,h,f, has big effects on efficiency
	#oceanData = readTxtTabs(dataFile)
	#oceanPred = readTxt(predFile)
	#dataDict = bucketOceans(m,oceanData)
	#predDict = bucketOceans(m,oceanPred)

	f,axarr = plt.subplot(5,4)
	i = 0
	j = 0
	for ocean in oceans[:4]:
		
		#Plot data
		rate = [r[1] for r in dataDict[ocean].iteritems()]
		colorsi = [colors[int(k)] for k in rate]
		axarr[i,j].barh(y_pos,rate,align='center',alpha=0.4,color=colorsi)
		axarr[i,j].yticks(y_pos,classifications)
		axarr[i,j].xlabel('Rate')
		axarr[i,j].title(ocean+', data')

		#Plot predictions
		rate = [r[1] for r in predDict[ocean].iteritems()]
		axarr[i,j+1].barh(y_pos,rate,align='center',alpha=0.4,color=colors)
		axarr[i,j+1].yticks(y_pos,classifications)
		axarr[i,j+1].xlabel('Rate')
		axarr[i,j+1].title(ocean+', predictions')
		print ocean

		i += 1

	i = 0
	j = 2
	for ocean in oceans[4:]:
		
		#Plot data
		rate = [r[1] for r in dataDict[ocean].iteritems()]
		axarr[i,j].barh(y_pos,rate,align='center',alpha=0.4,color=colors)
		axarr[i,j].yticks(y_pos,classifications)
		axarr[i,j].xlabel('Rate')
		axarr[i,j].title(ocean+', data')

		#Plot predictions
		rate = [r[1] for r in predDict[ocean].iteritems()]
		axarr[i,j+1].barh(y_pos,rate,align='center',alpha=0.4,color=colors)
		axarr[i,j+1].yticks(y_pos,classifications)
		axarr[i,j+1].xlabel('Rate')
		axarr[i,j+1].title(ocean+', predictions')
		print ocean

		i += 1

	plt.show()

'''
Stacked area chart to look at correlation between latitudes and classifications

'''

def ratesInBuckets(buckets):

	rates = {}
	for bucket in buckets:
		rates[bucket] = ratesOfPredictions([buckets[bucket]])
	return rates

def latBucketing(rawData,dx):

	buckets = {-90 + i*dx:[] for i in range(180/dx+1)}
	for bucket in buckets:
		for i in range(len(rawData['lon'])):
			x = rawData['lon'][i]
			y = rawData['lat'][i]
			if bucket + dx > y >= bucket:
				if not Basemap.is_land(m,m(x,y)[0],m(x,y)[1]):
					buckets[bucket].append(rawData['classif'][i])

	rates = ratesInBuckets(buckets)

	return rates

def sumRates(rate):

	total = 0
	for key,val in rate.iteritems():
		total += val
	return total

def graphVector(rates,dx):

	classifNr = {i:[] for i in np.arange(1.,14.)}
	y = np.arange(-90,90+dx,dx)

	for cl in classifNr:
		for i in y:
			print i
			sumR = sumRates(rates[int(i)])
			if cl in rates[int(i)]:
				classifNr[cl].append(rates[int(i)][cl]/float(sumR))
			else:
				classifNr[cl].append(0)

	return classifNr

def plotLats(classifNr,dx):

	x = [classifNr[i] for i in classifNr]
	n = 180/dx + 1
	y = np.arange(-90,90+dx,dx)
	fig = plt.figure()
	plt.stackplot(y,x,alpha = 0.4,colors = colors)
	plt.margins(0,0)
	plt.title('Classifications and latitudes, percentage')
	plt.xlabel('Latitudes')
	plt.xticks(y)
	plt.show()

def runLats(filename,dx):

	rawData = readTxt(filename)
	rates = latBucketing(rawData,dx)
	classifNr = graphVector(rates,dx)
	plotLats(classifNr,dx)



'''
CaCO3 comparison
'''

def CaPoints():

	lon = np.arange(-180.,181)
	lat = np.arange(-90.,91)

	return lon,lat

def findPred(filename,compareFile,rawDataPred):

	classif = []
	caco3 = []
	#rawDataPred = readTxt(filename)
	rawDataCaCO = readTxt(compareFile)
	for i in range(len(rawDataCaCO['lon'])):
		xi = rawDataCaCO['lon'][i]
		yi = rawDataCaCO['lat'][i]
		#Search for corresponding data point(possibly refine this later)
		for j in range(len(rawDataPred['lon'])):
			xj = rawDataPred['lon'][j]
			yj = rawDataPred['lat'][j]
			if xj > xi:
				break
			if xi == xj and yi == yj:
				classif.append(rawDataPred['classif'][j])
				caco3.append(rawDataCaCO['classif'][i])
				print 'found'
		print float(i)/len(rawDataCaCO['lon'])
	return classif,caco3

def plotCa(classif,caco3):

	plt.plot(classif,caco3,'.',alpha = 0.4)
	plt.show()

def plotCaRange(classif,caco3,CaRange):

	pltClassif = []
	pltCaco3 = []
	for i in range(len(caco3)):
		if CaRange[0] < caco3[i] < CaRange[1]:
			pltClassif.append(classif[i])
			pltCaco3.append(caco3[i])
	plotCa(pltClassif,pltCaco3)

def boxplotCa(classif,caco3):

	data = [[] for i in range(int(max(classif)))]
	print len(data)
	for i in range(len(caco3)):
		cl = int(classif[i]) - 1
		data[cl].append(caco3[i])

	fig = plt.figure(1,figsize = (9,6))
	ax = fig.add_subplot(111)
	boxes = ax.boxplot(data,patch_artist = True)
	for box in range(len(boxes['boxes'])):
		boxes['boxes'][box].set(color = colors[box],alpha = 0.4)
		boxes['boxes'][box].set(facecolor = colors[box],alpha = 0.4)
	
	for whisker in boxes['whiskers']:
		whisker.set(color = 'black' ,alpha = 0.4)

	for cap in boxes['caps']:
		cap.set(color = 'black',alpha = 0.4)

	for median in boxes['medians']:
		median.set(color = 'black', alpha = 0.4)

	for flier in boxes['fliers']:
		flier.set(marker = 'o',color = 'black',alpha = 0.4)

	#ax.set_title('CaCO3 content in different sediments')
	ax.set_ylabel('Ocean productivity [mgC / m**2 / day]')
	ax.set_xticklabels(classifications)
	plt.show()
import math

def updateCaData(filename = 'seabed_lithology_finegrid.txt'):

	f = open(filename)
	g = open(filename[:-4]+'noNaN.txt','w')
	lines = f.readlines()
	for line in lines:
		splitLine = re.split(r'\t+',line.strip())
		if splitLine[2] != 'NaN':	
			g.write(str(math.floor(float(splitLine[0])-180))+' '+str(math.floor(float(splitLine[1])))+' '+splitLine[2]+'\n')
	g.close()


def plotBathymetry(rawDataPred,bathFile = 'ETOPO1_Bed_g_gmt4.grd'):

	bathData = netcdf(bathFile,'r')
	bx = bathData.variables['x'][::60]
	by = bathData.variables['y'][::60]
	bz = bathData.variables['z'][::60,::60]

	data = []

	for i in range(len(rawDataPred['classif'])):
		xd = rawDataPred['lon'][i]
		yd = rawDataPred['lat'][i]
		for yi in itertools.ifilter(lambda m: abs(by[m] - yd) < 1e-5,range(len(by))):
			for xi in itertools.ifilter(lambda m: abs(bx[m] - xd) < 1e-5,range(len(bx))):
				if isOcean(xd,yd):
					data.append([rawDataPred['classif'][i],bz[yi][xi]])
		print float(i)/len(rawDataPred['classif'])
	return data



def filterFun(xi,yi,x,y):
	if abs(xi - x) < 1e-5 and abs(yi - y) < 1e-5:
		return True
	else:
		return False

def isOcean(x,y):

	return not Basemap.is_land(m,m(x,y)[0],m(x,y)[1])

def plotT(rawDataPred,TFile = 'temperaturenoNaN.txt'):

	rDT = readTxt(TFile)
	data = []
	for i in range(len(rawDataPred['classif'])):
		xd = rawDataPred['lon'][i]
		yd = rawDataPred['lat'][i]
		count = 0
		for j in itertools.ifilter(lambda m: abs(rDT['lat'][m] - yd) < 1e-5 and abs(rDT['lon'][m] - xd) <1e-5,range(len(rDT['lat']))):
			print 'made it'
			if isOcean(xd,yd):
				data.append([rawDataPred['classif'][i],rDT['classif'][j]])
				count += 1
		if count > 1:
			print 'something wrong'
		if not count:
			print 'not used'
	return data

'''
Looking at bio activity
'''

def cleanNoPoints(filename):

	f = open(filename)
	g = open(filename[:-4]+'upd.txt','w')
	lines = f.readlines()
	for line in lines:
		splitLine = re.split(r'\t+',line.strip())[0].split()
		if splitLine[2] != '-9999':
			g.write(splitLine[0]+' '+splitLine[1]+' '+splitLine[2]+'\n')
	g.close()












































