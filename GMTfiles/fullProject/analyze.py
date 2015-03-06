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
import os
import copy

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

def readTxt(filename,xyCheck,zCheck,comment,nans):

	d = {}
	d['lon'],d['lat'],d['classif'] = np.loadtxt(filename,comments = comment).T
	data = filterLi(d['lon'],d['lat'],d['classif'],xyCheck,zCheck)
	return data

def filterLi(x,y,z,xyCheck,zCheck):

	data = {}
	data = {'lon':[],'lat':[],'classif':[]}
	for xi,yi,zi in zip(x,y,z):
		if xyCheck(xi,yi) and zCheck(zi):
			data['lon'].append(xi)
			data['lat'].append(yi)
			data['classif'].append(zi)

	return data

def readNetcdf(ncFile,ptsPerDeg,xyCheck,zCheck):

	'''
	Reads in a netcdf file that has ptsPerDeg points per degree, 0.1x0.1 degree grid has ptsPerDeg=10
	'''

	ncData = Dataset(ncFile,'r')
	nx = ncData.variables['lon'][:]
	ny = ncData.variables['lat'][:]
	nz = ncData.variables['z'][:][:,:]

	return loopReadFile(nx,ny,nz,xyCheck,zCheck)

def loopReadFile(nx,ny,nz,xyCheck,zCheck):

	data = {'lon':[],'lat':[],'classif':[]}

	for i,j in itertools.product(enumerate(nx),enumerate(ny)):
		xi = rollOff2pi(i[1])
		yj = j[1]
		zij= nz[j[0],i[0]]
		if xyCheck(xi,yj) and zCheck(zij):	
			data['lon'].append(xi)
			data['lat'].append(yj)
			data['classif'].append(zij)
	return data


def rollOff2pi(xi):

	while not -180 < xi < 180:
		xi = xi - np.sign(xi) * 360
	return xi


'''
Boxplot functions
'''


def mapXY(x,y,x0,y0,yn):
	return (yn - y0 + 1) * (x - x0) + y - y0

def emptyVec(vectorLen):

	vec1 = np.empty((vectorLen,1))
	vec2 = np.empty((vectorLen,1))
	vec1[:] = np.NaN
	vec2[:] = np.NaN

	return flatten(vec1),flatten(vec2)

def flatten(l):
	return [i for sublist in l for i in sublist]

def oceanCheck(x,y,o):


	if o == 'natl':
		return ocean(x,y) == o or ocean(x,y) == 'catl'
	if o == 'npac':
		return ocean(x,y) == o or ocean(x,y) == 'cpac'
	else:
		return ocean(x,y) == o

def fillList(li,rawData,x0,y0,yn):

	for i,j in enumerate(rawData['classif']):

		xi = rawData['lon'][i]
		yi = rawData['lat'][i]
		ind = mapXY(xi,yi,x0,y0,yn)
		li[ind] = j

	return li

def cleanNans(vec1,vec2):

		ve1,ve2 = [],[]

		for i,j in zip(vec1,vec2):
			if not np.isnan([i,j]).any():
				ve1.append(i)
				ve2.append(j)
		return ve1,ve2

oceans = {'natl':[0,0],'satl':[0,1],'nind':[1,0],'sind':[1,1],'npac':[2,0],'spac':[2,1],'arctic':[3,0],'antarctic':[3,1]}

def histCaPred(file1,file2,region,zCheck1,zCheck2):

	x0,xn,y0,yn = region
	vectorLen = (xn - x0 + 1) * (yn - y0 + 1)
	for o,v in oceans.iteritems():

		vec1,vec2 = emptyVec(vectorLen)
		data1 = readNetcdf(file1,1,lambda x,y:oceanCheck(x,y,o),zCheck1)
		data2 = readNetcdf(file2,1,lambda x,y:oceanCheck(x,y,o),zCheck2)
		vec1 = fillList(vec1,data1,x0,y0,yn)
		vec2 = fillList(vec2,data2,x0,y0,yn)
		vec1,vec2 = cleanNans(vec1,vec2)
		boxplotCa(vec1,vec2,'CaCO3 %s' % (o),'CaCO3%s.ps' % (o),'ps',600)


def boxplotCa(classif,caco3,ytext,plotname,plotformat,res):

	data = sortClassif(classif,caco3)
	
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

	plt.savefig(plotname,format = plotformat,dpi = res)
	plt.savefig(plotname[:-3]+'.png',format = 'png',dpi = res)
	plt.clf()



def plotElem(rawDataPred,rDT,xyCheck ,zCheck ):

	d = pd.DataFrame(rawDataPred)
	d2= pd.DataFrame(rDT)
	d3= d.merge(d2,left_on = ['lon','lat'],right_on = ['lon','lat'])

	return d3.classif_x , d3.classif_y



elements 	= ['nitrate','silicate','phosphate']
elemLabel	= {'nitrate' : 'n' , 'silicate' : 'i' , 'phosphate' : 'p'}
seasons 	= ['Annual']
seasonLabel	= {'Winter' : '13', 'Summer' : '15' , 'Annual' : '00'}

files = ['seabed_lithology_ant.nc','seabed_lithology_nind.nc', 'seabed_lithology_satl.nc', 'seabed_lithology_sind.nc', 'seabed_lithology_spac.nc']#'seabed_lithology_arctic.nc', 
jointFiles = ['seabed_lithology_catl.nc', 'seabed_lithology_cpac.nc', 'seabed_lithology_natl.nc', 'seabed_lithology_npac.nc']	

def oceanPlots():

	files = ['seabed_lithology_ant.nc', 'seabed_lithology_arctic.nc', 'seabed_lithology_nind.nc', 'seabed_lithology_satl.nc', 'seabed_lithology_sind.nc', 'seabed_lithology_spac.nc']
 	jointFiles = ['seabed_lithology_catl.nc', 'seabed_lithology_cpac.nc', 'seabed_lithology_natl.nc', 'seabed_lithology_npac.nc']
	bathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',1,noCheck,lambda bz : bz < 0)

	for f in files:
		rawData = readNetcdf('regions/%s' % (f),1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
		rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
		boxplotCa(rcl,rba,'Bathymetry %s [m]' % (f[-7:-3]),'gmtplots/bathymetry%s.ps' % (f[-7:-3]),'ps',600)

	rawData = readNetcdf('regions/seabed_lithology_catl.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData2= readNetcdf('regions/seabed_lithology_natl.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	boxplotCa(rcl,rba,'Bathymetry %s [m]' % ('natl and catl'),'gmtplots/bathymetry%s.ps' % ('natlandcatl') ,'ps',600)

	rawData = readNetcdf('regions/seabed_lithology_cpac.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData2= readNetcdf('regions/seabed_lithology_npac.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	boxplotCa(rcl,rba,'Bathymetry %s [m]' % ('npac and cpac'),'gmtplots/bathymetry%s.ps' % ('npacandcpac') ,'ps',600)


def histOceans(binNr,cl,colrs,plotname,res):

	bathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',1,noCheck,lambda bz : bz < 0)
	for f in files:
		rawData = readNetcdf('regions/%s' % (f),1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
		rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
		histograms(rba,binNr,colrs,'Bathymetry %s [m]' % (f[-7:-3]),'bathymetry%s%sbins' % (f[-7:-3],binNr),res)

	rawData = readNetcdf('regions/seabed_lithology_catl.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData2= readNetcdf('regions/seabed_lithology_natl.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	histograms(rba,binNr,colrs,'Bathymetry %s [m]' % ('natl and catl'),'bathymetryNatlCatl%sbins'%(binNr),res)

	rawData = readNetcdf('regions/seabed_lithology_cpac.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData2= readNetcdf('regions/seabed_lithology_npac.nc',1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	histograms(rba,binNr,colrs,'Bathymetry %s [m]' % ('npac and cpac'),'bathymetryNpacCpac%sbins'%(binNr),res) 



def histData(filename,cl):
	return readNetcdf(filename,1,lambda x,y:x%1==0 and y%1==0,lambda z:notNan(z) and z == cl)

def histograms(data,binNr,colrs,ylab,plotname,res):

	plt.hist(data,bins=binNr,color=colrs)
	plt.ylabel(ylab)
	plt.tick_params(axis='both',direction='outward',labeltop='off',labelright='off',color = colrs)
	plt.savefig(plotname+'.ps',format = 'ps',dpi = res)
	plt.savefig(plotname+'.png',format = 'png',dpi = res)
	plt.close()







'''
Helper functions
'''

def noCheck(*args):
	return True

def nonNegativity(z):
	return z >= 0

def notNan(z):
	return not np.isnan(z) and not type(z) == np.ma.core.MaskedConstant

def zRange(z):
	return zmin < z < zmax




'''
klara cpt

opna allar excel skrar
finna hvar whiskers eru 2.5*q1 - 1.5*q3 og 2.5*q3 - 1.5*q1
'''

def cptRanges():

	h = 'gmtplots/'
	files = {h+'boxplots/Shared/bathBoxplot.xlsx':h+'netcdffiles/bathymetryMasked.nc',h+'boxplots/Shared/oxygenBoxplots.xlsx':h+'netcdffiles/oxygen.nc',h+'boxplots/prodBoxplotWinter.xlsx':h+'productivity/winterAll.nc',h+'boxplots/prodBoxplotSummer.xlsx':h+'productivity/summerAll.nc',h+'boxplots/Shared/salinBoxplot.xlsx':h+'netcdffiles/salinity.nc',h+'boxplots/Shared/Tboxplot.xlsx':h+'netcdffiles/temperature.nc'}

	for File in files:

		f = pd.io.excel.read_excel(File)
		fmin,fmax = min(2.5 * f.irow(1) - 1.5 * f.irow(3)),max(2.5 * f.irow(3) - 1.5 * f.irow(1))
		print fmin,fmax
		fmin = max(fmin,min(f.irow(0)))
		fmax = min(fmax,max(f.irow(4)))
		print fmin,fmax
		if File in [h+'boxplots/Shared/oxygenBoxplots.xlsx',h+'boxplots/prodBoxplotWinter.xlsx',h+'boxplots/prodBoxplotSummer.xlsx',h+'boxplots/Shared/salinBoxplot.xlsx']:
			if fmin < 0:
				fmin = 0
		os.system('gmt grd2cpt %s -L%f/%f -D > %s.cpt' % (files[File],fmin,fmax,files[File][:-3]))	





def maskMaps():

	seasons = ['winter','summer']
	years = [str(i) for i in range(2002,2015)]
	for year in years:
		if year != '2002':
			for season in seasons:
				filename = 'gmtplots/productivity/'+season+year
				if season is 'winter':
					os.system('gmt grdsample '+filename+'.nc -Gtest.nc -I1 -R-180/180/-89/0')
				else:
					os.system('gmt grdsample '+filename+'.nc -Gtest.nc -I1 -R-180/180/0/89')
				os.system('gmt grdmath test.nc gmtplots/productivity/'+season+'mask.nc OR = '+filename+'Masked.nc ')
		else:
			season = 'summer'
			filename = 'gmtplots/productivity/'+season+year
			os.system('gmt grdsample '+filename+'.nc -Gtest.nc -I1 -R-180/180/0/89')
			os.system('gmt grdmath test.nc gmtplots/productivity/'+season+'mask.nc OR = '+filename+'Masked.nc ')



def fullNutrMaps():


	for element in elements:
		for season in seasons:
			filename = 'Nutrients/%s%s/%s%san1.001' % (element,season,elemLabel[element],seasonLabel[season])
			cleanFile= filename + 'clean'
			ncFile = filename + '.nc'
			clean(filename,cleanFile)
			prearea = '-179.5/179.5/-89.5/89.5'
			#mismunandi kort fyrir mismunandi season, -R og mask
			if season == 'Summer':
				area = '-180/180/0/89'
				mask = 'summermask.nc'
			if season == 'Winter':
				area = '-180/180/-89/0'
				mask = 'wintermask.nc'
			if season == 'Annual':
				area = '-180/180/-89/89'
				mask = 'seabed_lithology_finegrid_weighted_ocean_v61deg.nc'

			#laga area og allt thetta
			os.system('gmt xyz2grd %s -G%s -I1 -R%s -fg -:' % (cleanFile,ncFile,prearea))
			os.system('gmt grdsample %s -G%s -I1 -R%s -fg' % (ncFile,ncFile,area))
			os.system('gmt grdmath %s %s OR = %s' % (ncFile,mask,ncFile))

def fullProdMaps():

	leapYears = ['2004','2008','2012']
	years= [str(year) for year in range(2003,2014) if year not in leapYears]
	days = ['001','032','060','091','121','152','182', '213', '244', '274', '305','335']
	leapDays = ['001','032','061','092','122','153','183', '214', '245', '275', '306','336']
	maps = {year:days for year in years}
	leapMaps = {year:leapDays for year in leapYears}
	borderMaps = {'2002':days[6:],'2014':days[:-1]}
	allMaps = dict(maps.items()+leapMaps.items()+borderMaps.items())
	header = lambda m,s:'gmtplots/productivity/'+year+'/'+year+m+s+'.nc '
	headers = lambda m,s:header(m[0],s)+header(m[1],s)+'ADD '+header(m[2],s)

	for year in allMaps:
		if year != '2002':
			winter = allMaps[year][0:3]
			summer = allMaps[year][6:9]
			for month in winter:
				os.system('gmt surface gmtplots/productivity/'+year+'/vgpm.'+year+month+'.all.xyzclean -Ggmtplots/productivity/'+year+'/'+year+month+'winter.nc -R-180/180/-89/89 -I1 -fg -Ll0')
			for month in summer:
				os.system('gmt surface gmtplots/productivity/'+year+'/vgpm.'+year+month+'.all.xyzclean -Ggmtplots/productivity/'+year+'/'+year+month+'summer.nc -R-180/180/-89/89 -I1 -fg -Ll0')
			os.system('gmt grdmath '+headers(winter,'winter')+'ADD 0.33333 MUL = gmtplots/productivity/winter'+year+'.nc')
			os.system('gmt grdmath '+headers(summer,'summer')+'ADD 0.33333 MUL = gmtplots/productivity/summer'+year+'.nc')
		else:
			summer = allMaps[year][0:3]
			for month in summer:
				os.system('gmt surface gmtplots/productivity/'+year+'/vgpm.'+year+month+'.all.xyzclean -Ggmtplots/productivity/'+year+'/'+year+month+'summer.nc -R-180/180/-89/89 -I1 -fg -Ll0')
			os.system('gmt grdmath '+headers(summer,'summer')+'ADD 0.33333 MUL = gmtplots/productivity/summer'+year+'.nc')


def multivarAnalysis():

	x = np.arange(-180,181)
	y = np.arange(-89,90)
	xvec = [xi for xi in x for yi in y]
	yvec = [yi for xi in x for yi in y]
	sali = [np.nan for i in xvec]
	temp = copy.copy(sali)
	sili = copy.copy(sali)
	nitr = copy.copy(sali)
	phos = copy.copy(sali)
	bath = copy.copy(sali)
	prod = copy.copy(sali)
	sali = fillList(sali,readNetcdf('gmtplots/netcdffiles/salinity.nc',['lon','lat','z'],1,noCheck,notNan))
	temp = fillList(temp,readNetcdf('gmtplots/netcdffiles/temperature.nc',['lon','lat','z'],1,noCheck,notNan))
	sili = fillList(sili,readNetcdf('gmtplots/netcdffiles/silicateAnnual.nc',['lon','lat','z'],1,noCheck,notNan))
	nitr = fillList(nitr,readNetcdf('gmtplots/netcdffiles/nitrateAnnual.nc',['lon','lat','z'],1,noCheck,notNan))
	phos = fillList(phos,readNetcdf('gmtplots/netcdffiles/phosphateAnnual.nc',['lon','lat','z'],1,noCheck,notNan))
	bath = fillList(bath,readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',['lon','lat','z'],1,noCheck,notNan))
	prodSummer = readNetcdf('gmtplots/netcdffiles/summerAll.nc',['lon','lat','z'],1,noCheck,notNan)
	prodWinter = readNetcdf('gmtplots/netcdffiles/winterAll.nc',['lon','lat','z'],1,noCheck,notNan)
	prodSummer['classif'].extend(prodWinter['classif'])
	prodSummer['lon'].extend(prodWinter['lon'])
	prodSummer['lat'].extend(prodWinter['lat'])
	prod = fillList(prod,prodSummer)
	#ekki prenta ef ekkert i listanum
	stats = DataFrame([[i+1,xvec[i],yvec[i],sali[i],temp[i],sili[i],nitr[i],phos[i],bath[i],prod[i]] for i in range(len(xvec)) if not np.isnan([i+1,xvec[i],yvec[i],sali[i],temp[i],sili[i],nitr[i],phos[i],bath[i],prod[i]]).any()], index = None , columns = None)
	stats.to_csv('lithologyStats.m',sep = '\t', index = None , columns = None)







def ocean(x,y):

	if -100 <= x <= 20 and 20 <= y <= 65:
		return 'natl'
	if -80 <= x <= 20 and 0 <= y <= 20:
		return 'catl'
	if -70 <= x <= 20 and -40 <= y <= 0:
		return 'satl'
	if 20 <= x <= 120 and 0 <= y <= 24.5:
		return 'nind'
	if 20 <= x <= 120 and -40 <= y <= 0:
		return 'sind'
	if 120 <= x <= 260 and 20 <= y <= 59.5:
		return 'npac'
	if (120 <= x <= 180 or -180 <= x <= -100) and  20 <= y <= 59.5:
		return 'npac'
	if 120 <= x <= 280 and 0 <= y <= 20:
		return 'cpac'
	if 120 <= x <= 290 and -40 <= y <= 0:
		return 'spac'
	if (120 <= x <= 180 or -180 <= x <= -80) and 0 <= y <= 20:
		return 'cpac'
	if (120 <= x <= 180 or -180 <= x <= -70) and -40 <= y <= 0:
		return 'spac'
	if y > 65:
		return 'arctic'
	if y < -40:
		return 'ant'
	return 'none'


def pointsInOcean():

	oceans = ['natl','satl','nind','sind','npac','spac','antarctic']
	for o in oceans:		
		for i in [20,30,40,50,60]:

			data = readNetcdf('contourCaCO3/caco3%s%s.nc' % (i,o),1,lambda x,y:oceanCheck(x,y,o),notNan)
			print 'There are %s points in %s for the %s contour file' % (str(np.count_nonzero(data['classif'])),o,str(i))
			if np.count_nonzero(data['classif']) < 20:
				print data['lon'],data['lat']
				plt.clf()
				plt.plot(data['lon'],data['lat'],'.')
				plt.show()





def histCaOceans(caco3,binNr):

	oceans = ['natl','satl','nind','sind','npac','spac','antarctic']
	conData = readNetcdf('test%s.nc' % (caco3),['lon','lat','z'],1,noCheck,notNan)

	for o in oceans:

		bathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',1,oceanCheck,notNan)
		ccl,cba = plotElem(conData,bathData,noCheck,noCheck)
		if cba:
			histograms(cba,binNr,colors[4],'Bathymetry at CaCO3 %s %s' % (caco3,o),'caco3bath%s%sbins%sperc' % (o,binNr,caco3) ,600)

def dotgramsCaOceans(caco3,binSpace):

	oceans = {'natl':[0,0],'satl':[0,1],'nind':[1,0],'sind':[1,1],'npac':[2,0],'spac':[2,1],'arctic':[3,0],'antarctic':[3,1]}
	
	f,axarr = plt.subplots(4,2)

	for o,v in oceans.iteritems():
		print o
		conData = readNetcdf('CCD/CaCO3contour%s.nc' % (caco3),1,lambda x,y:oceanCheck(x,y,o),notNan)
		bathData = readNetcdf('CCD/bathymetry01deg.nc',1,lambda x,y:oceanCheck(x,y,o),lambda z:notNan(z) and not nonNegativity(z))
		ccl,cba = plotElem(conData,bathData,noCheck,noCheck)
		if len(cba) < 2:
			cba = [-1,-1,-1]
			bins = 3
		else:
			binsMin = np.floor(np.min(cba) / float(binSpace)) * binSpace
			binsMax = np.ceil(np.max(cba) / float(binSpace)) * binSpace
			bins = np.arange(binsMin,binsMax,binSpace)

		axarr[v[0]][v[1]].hist(cba,bins=bins,color = colors[4])
		axarr[v[0]][v[1]].set_title(o)

	plt.savefig('CCD/CCD%s%sbins.png' % (caco3,binSpace),format = 'png',dpi = 600)
	plt.show()
	plt.close()

def histCaAllOceans():

	oceans = {'natl':[0,0],'satl':[0,1],'nind':[1,0],'sind':[1,1],'npac':[2,0],'spac':[2,1],'arctic':[3,0],'antarctic':[3,1]}
	
	f,axarr = plt.subplots(4,2)


	for o,v in oceans.iteritems():
		cbaVec = []
		for caco3 in [20,30,40,50,60]:	
			if o == 'natl':
				oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'catl'
			if o == 'npac':
				oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'cpac'
			else:
				oceanCheck = lambda x,y: ocean(x,y) == o
			conData = readNetcdf('contourCaCO3/caco3%s%s.nc' % (caco3,o),1,noCheck,notNan)
			bathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',1,noCheck,lambda z:notNan(z) and not nonNegativity(z))
			print len(conData['classif']),len(bathData['classif'])
			ccl,cba = plotElem(conData,bathData,noCheck,noCheck)
			print len(cba)
			
			if not cba:
				cba = [0,0,0]
			cbaVec.append(cba)
		boxes = axarr[v[0]][v[1]].boxplot(cbaVec,0,'ko',0.4,patch_artist = True)
		for box in range(len(boxes['boxes'])):
			boxes['boxes'][box].set(color = colors[4],alpha = 0.4)
			boxes['boxes'][box].set(facecolor = colors[4],alpha = 0.4)
	
		for whisker in boxes['whiskers']:
			whisker.set(color = 'black' ,alpha = 0.4)

		for median in boxes['medians']:
			median.set(color = 'black', alpha = 0.4)

		for flier in boxes['fliers']:
			flier.set(marker = '.',color = 'black',alpha = 0.4)
		axarr[v[0]][v[1]].set_title(o)
		axarr[v[0]][v[1]].set_xticklabels(['20','30','40','50','60'])
	plt.savefig('oceanBoxplots.png',format = 'png',dpi = 600)
	plt.show()
	plt.close()



def piesAllOceans():

	oceans = {'natl':[0,0],'satl':[0,1],'nind':[1,0],'sind':[1,1],'npac':[2,0],'spac':[2,1],'arctic':[3,0],'ant':[3,1]}
	
	f,axarr = plt.subplots(4,2)

	for o,v in oceans.iteritems():
		if o == 'natl':
			oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'catl'
		if o == 'npac':
			oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'cpac'
		else:
			oceanCheck = lambda x,y: ocean(x,y) == o
		conData = readNetcdf('seabed_lithology_finegrid_weighted_ocean_v61deg.nc',1,oceanCheck,notNan)
		cbaVec = [conData['classif'].count(i) for i in range(1,14)]
		print cbaVec

		boxes = axarr[v[0]][v[1]].pie(cbaVec,colors = colors,labels = [str(i) for i in range(1,14)])
		axarr[v[0]][v[1]].axis('equal')
		axarr[v[0]][v[1]].set_title(o)
#	plt.savefig('oceanBoxplots.png',format = 'png',dpi = 600)
	plt.show()
	plt.close()

def histCaAllOceans():

	oceans = {'natl':[0,0],'satl':[0,1],'nind':[1,0],'sind':[1,1],'npac':[2,0],'spac':[2,1],'arctic':[3,0],'antarctic':[3,1]}
	
	f,axarr = plt.subplots(4,2)


	for o,v in oceans.iteritems():
		cbaVec = []
		for caco3 in [20,30,40,50,60]:	
			if o == 'natl':
				oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'catl'
			if o == 'npac':
				oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'cpac'
			else:
				oceanCheck = lambda x,y: ocean(x,y) == o
			conData = readNetcdf('CCD/CaCO3%s.nc' % (caco3),1,oceanCheck,notNan)
			bathData = readNetcdf('CCD/bathymetry01deg.nc',1,oceanCheck,lambda z:notNan(z) and not nonNegativity(z))
			print len(conData['classif']),len(bathData['classif'])
			ccl,cba = plotElem(conData,bathData,noCheck,noCheck)
			print len(cba)
			
			if not cba:
				cba = [0,0,0]
			cbaVec.append(cba)
		boxes = axarr[v[0]][v[1]].boxplot(cbaVec,0,'ko',0.4,patch_artist = True)
		for box in range(len(boxes['boxes'])):
			boxes['boxes'][box].set(color = colors[4],alpha = 0.4)
			boxes['boxes'][box].set(facecolor = colors[4],alpha = 0.4)
	
		for whisker in boxes['whiskers']:
			whisker.set(color = 'black' ,alpha = 0.4)

		for median in boxes['medians']:
			median.set(color = 'black', alpha = 0.4)

		for flier in boxes['fliers']:
			flier.set(marker = '.',color = 'black',alpha = 0.4)
		axarr[v[0]][v[1]].set_title(o)
		axarr[v[0]][v[1]].set_xticklabels(['20','30','40','50','60'])
	plt.savefig('oceanBoxplots.png',format = 'png',dpi = 600)
	plt.show()
	plt.close()


'''
Formatting .csv files
'''

years = [str(year) for year in range(1998,2008)]
months= ['01','02','03','04','05','06','07','08','09','10','11','12']
sizes = ['micro','nano','pico']
seasons={'winter':['01','02','03'],'spring':['04','05','06'],'summer':['07','08','09'],'fall':['10','11','12']}

def readCSV(filename):

	lines = open(filename).readlines()
	cleanData = [cleanLines(l) for l in lines]

	return cleanData

def cleanLines(line):

	l = line.split(';')
	l[-1] = l[-1][:-1]
	l = [np.nan if i == 'NA' else i for i in l]

	return l

def writeCSV(data,filename):

	x = np.linspace(-179.5,179.5,360)
	y = np.linspace(-89.5,89.5,180)
	with open(filename,'w') as g:
		for i,xi in enumerate(x):
			for j,yj in enumerate(y):
				g.write('%r %r %s\n' % (xi,yj,data[i][j]))

def fileExists(filename):

	return os.path.isfile(filename)

def loopCSV(folder):

	for year,month,size in itertools.product(years,months,sizes):
		filename = '%s/%s%s-%s.csv' % (folder,year,month,size)
		if fileExists(filename):
			newFile = filename[:-3] + 'txt'
			data = readCSV(filename)
			writeCSV(data,newFile)
			print '%s finished' % (filename)
		else:
			print '%s doesnt exist' % (filename)

def loopNetcdf(folder):

	for year,month,size in itertools.product(years,months,sizes):
		filename = '%s/%s%s-%s.txt' % (folder,year,month,size)
		if fileExists(filename):
			newFile = filename[:-3] + 'nc'
			fineGrid= filename[:-4] + '01deg.nc'
			os.system('gmt xyz2grd %s -G%s -I1 -R-179.5/179.5/-89.5/89.5 -fg' % (filename,newFile))
			os.system('gmt grdsample %s -G%s -I0.1 -R-180/180/-89/89 -fg' % (newFile,fineGrid))
			print '%s finished' % (filename)
		else:
			print '%s doesnt exist' % (filename)

def addSeasons(folder):

	for year in years:
		for season in seasons:
			for size in sizes:
				filenames = ['%s/%s%s-%s01deg.nc' % (folder,year,month,size) for month in seasons[season]]
				if all([fileExists(f) for f in filenames]):
					newFile = '%s/%s%s-%s.nc' % (folder,year,season,size)
					command = 'gmt grdmath %s %s ADD %s ADD 3 DIV = %s' % (filenames[0],filenames[1],filenames[2],newFile)
					os.system(command)


def boxProdScales(folder):

	plots = {'summer-micro':[0,0],'winter-micro':[0,1],'summer-nano':[1,0],'winter-nano':[1,1],'summer-pico':[2,0],'winter-pico':[2,1]}
	
	f,axarr = plt.subplots(3,2)

	data = readNetcdf('seabed_lithology_finegrid_weighted_ocean_v6.nc',1,noCheck,notNan)
	for o,v in plots.iteritems():
		data2 = readNetcdf('%s/%s.nc' % (folder,o),1,noCheck,lambda z:notNan(z) and nonNegativity(z))
		cl,ba = plotElem(data,data2,noCheck,noCheck)
		da = sortClassif(cl,ba)
		colorsBox = [colors[int(i) - 1] for i in sorted(set(cl))]
		print [len(d) for d in da]
		if not da:
			da = [-1,-1,-1]
		boxes = axarr[v[0]][v[1]].boxplot([d for d in da if d],0,'ko',0.4,patch_artist = True)
		for box in range(len(boxes['boxes'])):
			boxes['boxes'][box].set(color = colorsBox[box],alpha = 0.4)
			boxes['boxes'][box].set(facecolor = colorsBox[box],alpha = 0.4)
	
		for whisker in boxes['whiskers']:
			whisker.set(color = 'black' ,alpha = 0.4)

		for median in boxes['medians']:
			median.set(color = 'black', alpha = 0.4)

		for flier in boxes['fliers']:
			flier.set(marker = '.',color = 'black',alpha = 0.4)
		axarr[v[0]][v[1]].set_title(o)
		axarr[v[0]][v[1]].set_ylabel('Productivity [g C m**-2 d**-1]')
		axarr[v[0]][v[1]].set_xticklabels([str(int(i)) for i in sorted(set(cl))])
	plt.savefig('productivityBoxplots.png',format = 'png',dpi = 600)
	plt.show()
	plt.close()


def sortClassif(cl,ba):

	data = [[] for i in range(len(classifications))]
	for i,j in enumerate(cl):
		ind = int(j) - 1
		data[ind].append(ba[i])
	return data




























