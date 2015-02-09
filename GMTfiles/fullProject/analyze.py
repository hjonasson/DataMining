from mpl_toolkits.basemap import Basemap
from scipy.io import netcdf_file as netcdf
import numpy as np
import sklearn
import re
import itertools
import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import math
from pandas import DataFrame
import pandas as pd
from netCDF4 import Dataset


'''
Global constants
'''

classifications = ('Gravel','Sand','Silt','Silicious clay','Calc. ooze','Radiolarian ooze','Diatom ooze','Sponge ooze','Calc./sili. ooze','Shells and coral','Ash,glass,volcanics','Mud','Fine-grained calc.')
colors = ["#808284","#FFF100","#FAA918","#704B2A","#0E91CF","#0D9647","#BED753","#55938D","#8370B2","#F7BBD5","#D83A26","#C39A6B","#002EA7"]
m = Basemap(projection='merc',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,lat_ts=5,resolution='c')#res c,l,i,h,f, has big effects on efficiency


'''
Reading functions
'''

def clean(filename,newFile):

	f = open(filename)
	lines = f.readlines()
	g = open(newFile,'w')
	for line in lines:
		splitLine = re.split(r'\t+',line.strip())[0].split()
		if line != lines[0]:
			if splitLine[2] != '-99.999':
				g.write(splitLine[0] + ' ' + splitLine[1] + ' ' + splitLine[2] + '\n')
			else:
				g.write(splitLine[0] + ' ' + splitLine[1] + ' NaN\n')
	g.close()

def readTxt(filename,xyCheck , zCheck = lambda zi: zi != 'NaN'):

	'''
	Takes in a filename of a three column format and gives
	the data from it in a dictionary

	to get 1x1 degree grid: xyCheck = lambda x,y: not x % 1 and not y % 1
	'''

	data = {'lon':[],'lat':[],'classif':[]}
	f = open(filename)
	lines = f.readlines()
	for line in lines:
		if line[0] != '>' and line[1] != 'L':
			splitLine = re.split(r'\t+',line.strip())[0].split()
			if xyCheck(float(splitLine[1]),float(splitLine[0])) and zCheck(float(splitLine[2])):
				data['lon'].append(float(splitLine[1]))
				data['lat'].append(float(splitLine[0]))
				data['classif'].append(float(splitLine[2]))
	return data

def readNetcdf2(ncFile,keyList,ptsPerDeg,xyCheck,zCheck):

	'''
	Reads in a netcdf file that has ptsPerDeg points per degree, 0.1x0.1 degree grid has ptsPerDeg=10
	'''

	ncData = Dataset(ncFile,'r')
	nx = ncData.variables['lon'][::ptsPerDeg]
	ny = ncData.variables['lat'][::ptsPerDeg]
	nz = ncData.variables['z'][::ptsPerDeg,::ptsPerDeg]

	data = {'lon':[],'lat':[],'classif':[]}
	for i in range(len(nx)):
		for j in range(len(ny)):
			xi = nx[i]
			yj = ny[j]
			zij= nz[j,i]
			if xyCheck(xi,yj) and zCheck(zij):	
				data['lon'].append(xi)
				data['lat'].append(yj)
				data['classif'].append(zij)
	return data

def readNetcdf(ncFile,keyList,ptsPerDeg,xyCheck,zCheck):

	'''
	Reads in a netcdf file that has ptsPerDeg points per degree, 0.1x0.1 degree grid has ptsPerDeg=10
	'''

	ncData = Dataset(ncFile,'r')
	nx = ncData.variables['lon'][:]
	ny = ncData.variables['lat'][:]
	nz = ncData.variables['z'][:][:,:]

	data = {'lon':[],'lat':[],'classif':[]}
	for i in range(len(nx)):
		for j in range(len(ny)):
			xi = nx[i]
			yj = ny[j]
			zij= nz[j,i]
			if xyCheck(xi,yj) and zCheck(zij):	
				data['lon'].append(xi)
				data['lat'].append(yj)
				data['classif'].append(zij)
	return data

'''
Boxplot functions
'''



def boxplotCa(classif,caco3,ytext,plotname,plotformat,res):

	data = [[] for i in range(len(classifications))]
	for i in range(len(caco3)):
		cl = int(classif[i]) - 1
		data[cl].append(caco3[i])
	print len(data),[len(data[i]) for i in range(len(data))]
	
	# Write quartiles to csv
	colorsBox = [colors[int(i) - 1] for i in sorted(set(classif))]
	stats = np.transpose(np.array([np.percentile(data[int(i) - 1],[0,25,50,75,100]) for i in sorted(set(classif))]))
	statFrame = DataFrame(stats,index = ['0','25','50','75','100'],columns = [classifications[int(i) - 1] for i in sorted(set(classif))])
	writer = pd.ExcelWriter(plotname[:-3]+'.xlsx')
	statFrame.to_excel(writer,'Sheet1')
	writer.save()

	fig = plt.figure(1,figsize = (6,4))
	ax = fig.add_subplot(111)
	boxes = ax.boxplot([i for i in data if i],0,'ko',0.4,patch_artist = True)
	for box in range(len(boxes['boxes'])):
		boxes['boxes'][box].set(color = colorsBox[box],alpha = 0.4)
		boxes['boxes'][box].set(facecolor = colorsBox[box],alpha = 0.4)
	
	for whisker in boxes['whiskers']:
		whisker.set(color = 'black' ,alpha = 0.4)

	#for cap in boxes['caps']:
	#	cap.set(color = 'black',alpha = 0.4)

	for median in boxes['medians']:
		median.set(color = 'black', alpha = 0.4)

	for flier in boxes['fliers']:
		flier.set(marker = '.',color = 'black',alpha = 0.4)

	#ax.set_title('CaCO3 content in different sediments')
	ax.set_ylabel(ytext)
	ax.set_xticklabels([str(int(i)) for i in sorted(set(classif))])
	#Set ticks outwards
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.spines['left'].axis.axes.tick_params(direction = 'outward')

	#plt.savefig(plotname,format = plotformat,dpi = 600)
	plt.savefig(plotname[:-3]+'.png',format = 'png',dpi = res)
	plt.clf()

'''
Functions that compare two z values
'''

def plotNetcdf(rawDataPred,ncFile,ptsPerDeg,keyList,xyCheck,zCheck):

	'''
	rawDataPred is the rawData read from text file
	ncFile is the name of the netcdf file you want
	ptsPerDeg is 10 for 0.1deg grid
	keyList is a list of the keys in the netcdf file in ['x','y','z'] order

	for bathymetry run plotNetcdf(rawDataPred,'ETOPO1_Bed_g_gmt4.grd',60,['x','y','z'],isOcean,lambda bz: bz < 0)

	returns the z values parewise [[zi1,zj1],...]
	'''
	print keyList[0]
	ncData = Dataset(ncFile,'r')
	nx = ncData.variables[keyList[0]][::ptsPerDeg]
	ny = ncData.variables[keyList[1]][::ptsPerDeg]
	nz = ncData.variables[keyList[2]][:][::ptsPerDeg,::ptsPerDeg]

	data = []
	zlen = len(rawDataPred['classif'])
	nxlen = len(nx)
	nylen = len(ny)

	for i in range(zlen):
		xd = rawDataPred['lon'][i]
		yd = rawDataPred['lat'][i]
		if xyCheck(xd,yd):
			for yi in itertools.ifilter( lambda m:filterFun( int(ny[m]) , int(yd) ), range(nylen)):
				for xi in itertools.ifilter( lambda n:filterFun( int(nx[n]) , int(xd) ), range(nxlen)):
					zi = nz[yi,xi]
					if zCheck(zi):
						zd = rawDataPred['classif'][i]
						data.append([zd,zi])
		print float(i)/zlen

	return data



def plotElem(rawDataPred,rDT,xyCheck ,zCheck ):

	data = []
	zlen = len(rawDataPred['classif'])
	for i in range(zlen):
		xd = rawDataPred['lon'][i]
		yd = rawDataPred['lat'][i]
		for j in itertools.ifilter(lambda m: filterFun(rDT['lat'][m],yd) and filterFun(rDT['lon'][m],xd),range(len(rDT['lat']))):
			zj = rDT['classif'][j]
			if xyCheck(xd,yd) and zCheck(zj):
				data.append([rawDataPred['classif'][i],rDT['classif'][j]])
		print float(i)/zlen
	return [i[0] for i in data],[i[1] for i in data]



'''
Looking at nitrate, phosphate and silicate at different depths
'''

elements 	= ['nitrate','silicate','phosphate']
seasons 	= ['Winter'		, 'Summer']
hemispheres	= {'Winter' : 's', 'Summer': 'n'}


zmin = 30
zmax = 40

def allPlots(predFilename,TFile,salFile,oxyFile,res):
	
	'''
	Global classifications
	'''
	rawDataPred = readNetcdf2(predFilename,['lon','lat','z'],10,noCheck,notNan)
	
	'''
	Temperature boxplot 
	'''

	#rDT = readNetcdf(TFile,['lon','lat','z'],1,noCheck,notNan)
	#Tcl,Tba = plotElem(rawDataPred,rDT,noCheck,noCheck)
	#boxplotCa(Tcl,Tba,'Temperature [C]','gmtplots/boxplots/Tboxplot.ps','ps',res)
	
	'''
	Bathymetry boxplot 
	'''

	#BathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',60,['lon','lat','z'],noCheck,lambda bz : bz < 0)
	#print 'here'
	#Bcl,Bba = plotElem(rawDataPred,BathData,noCheck,noCheck)
	#print 'again'
	#boxplotCa(Bcl,Bba,'Bathymetry [m]','gmtplots/boxplots/bathBoxplot.ps','ps',res)
	
	'''
	Salinity boxplot 
	'''

	#SalData = readNetcdf(salFile,['lon','lat','z'],1,noCheck,notNan)
	#Scl,Sba = plotElem(rawDataPred,SalData,noCheck,noCheck)
	#boxplotCa(Scl,Sba,'Salinity','gmtplots/boxplots/salinBoxplot.ps','ps',res)
	

	'''
	Salinity boxplot with a tighter range
	'''

	#SalData = readNetcdf(salFile,['lon','lat','z'],1,noCheck,lambda z:nonNegativity(z) and zmin < z < zmax)
	#Scl,Sba = plotElem(rawDataPred,SalData,noCheck,noCheck)
	#boxplotCa(Scl,Sba,'Salinity','gmtplots/boxplots/salinBoxplot'+str(zmin)+str(zmax)+'.ps','ps',res)
	
	
	'''
	Dissolved oxygen boxplot
	'''

	OxyData = readNetcdf(oxyFile,['lon','lat','z'],1,noCheck,)
	Ocl,Oba = plotElem(rawDataPred,OxyData,noCheck,noCheck)
	boxplotCa(Ocl,Oba,'Dissolved oxygen','gmtplots/boxplots/oxygenBoxplot.ps','ps',res)
	
	
	'''
	Next look at summer values at each hemisphere
	'''

	#for season in seasons:

	#	'''
	#	Read only the summer hemisphere
	#	'''

	#	if season == 'Winter':
	#		latCheck = lambda x,y: not nonNegativity(y)
	#		bioFile = 'gmtplots/netcdffiles/productivityWinterKrigingMasked.nc'
	#	if season == 'Summer':
	#		latCheck = lambda x,y:nonNegativity(y)
	#		bioFile = 'gmtplots/netcdffiles/productivitySummerKrigingMasked.nc'
	#	rawDataPred = readNetcdf(predFilename,['lon','lat','z'],10,latCheck,notNan)

		
	#	'''
	#	Productivity boxplot done
	#	'''		
		
	#	bioData = readNetcdf(bioFile,['lon','lat','z'],1,latCheck,nonNegativity)
	#	biocl,bioba = plotElem(rawDataPred,bioData,noCheck,noCheck)
	#	boxplotCa(biocl,bioba,'Productivity '+season+' [mgC/m**2/day]','gmtplots/boxplots/prodBoxplot'+season+'.ps','ps',res)
		



'''
Helper functions
'''

def filterFun(xi,x):
	if abs(xi - x) < 1e-5: return True
	else: return False

def flooring(data):

	data['lon'] = floorEach(data['lon'])
	data['lat'] = floorEach(data['lat'])
	return data

def floorEach(dataList):
	return [i-.5 for i in dataList]

def isOcean(x,y):
	return not Basemap.is_land(m,m(x,y)[0],m(x,y)[1])

def noCheck(*args):
	return True

def nonNegativity(z):
	return z >= 0

def notNan(z):
	return not np.isnan(z) and not type(z) == np.ma.core.MaskedConstant

def zRange(z):
	return zmin < z < zmax


from sklearn.gaussian_process import GaussianProcess

def runKriging(filename,outFilename):

	rawData = readNetcdf(filename,['lon','lat','z'],1,noCheck,notNan)
	xpoints = np.arange(-180,180)
	ypoints = np.arange(-89,89)
	mapped = kriging(rawData,xpoints,ypoints)
	g = open(outFilename,'w')
	for i,xi in enumerate(xpoints):
		for j,yj in enumerate(ypoints):
			g.write(str(xi) + ' '+str(yi) + ' ' + str(mapped[i,j]))
	g.close()

def kriging(rawData,xpoints,ypoints):
	gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.01)
	data = [(rawData['lon'][i] , rawData['lat'][i]) for i in range(len(rawData['lon']))]
	labels = rawData['classif']
	print 'Fitting data to classifier'
	gp.fit(data,labels)
	print 'Finished fitting data'
	mapped = []
	for i,xi in enumerate(xpoints):
		yi = []
		for yj in ypoints:
			yi.append(gp.predict([xi,yj]))
		mapped.append(yi)
		print float(i)/len(xpoints)
	return mapped

def cleanProd():

	years = ['2004','2008','2012']
	days = ['001', '032', '061', '092', '122', '153', '183', '214', '245', '275', '306', '336']
	for year in years:
		print year
		for day in days:
			print day
			filename = 'gmtplots/productivity/' + year + '/vgpm.' + year + day + '.all.xyz'	
			f = open(filename)
			g = open(filename+'clean','w')
			lines = f.readlines()
			for line in lines:
				if line[0] != 'l':
					splitLine = re.split(r'\t+',line.strip())[0].split()
					if splitLine[2] != '-9999':
						g.write(line)
			g.close()



'''
klara cpt

opna allar excel skrar
finna hvar whiskers eru 2.5*q1 - 1.5*q3 og 2.5*q3 - 1.5*q1
'''

def cptRanges(*files):

	for File in files:

		f = pd.io.excel.read_excel(File)
































































