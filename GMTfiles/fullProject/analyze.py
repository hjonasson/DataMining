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

	plt.savefig(plotname,format = plotformat,dpi = res)
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



zmin = 30
zmax = 40

def allPlots(predFilename,TFile,salFile,oxyFile,res):
	print 'starting'	
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

	#OxyData = readNetcdf(oxyFile,['lon','lat','z'],1,noCheck,noCheck)
	#Ocl,Oba = plotElem(rawDataPred,OxyData,noCheck,noCheck)
	#boxplotCa(Ocl,Oba,'Dissolved oxygen','gmtplots/boxplots/oxygenBoxplot.ps','ps',res)
	
	
	#'''
	#Next look at summer values at each hemisphere
	#'''

	for season in seasons:

		'''
		Read only the summer hemisphere
		'''

		if season == 'Winter':
			latCheck = lambda x,y: not nonNegativity(y) and x % 1 == 0 and y % 1 == 0
	#		bioFile = 'gmtplots/productivity/winterAll.nc'
	#		clip = 800
		if season == 'Summer':
			latCheck = lambda x,y:nonNegativity(y) and x % 1 == 0 and y % 1 == 0
	#		bioFile = 'gmtplots/productivity/summerAll.nc'
	#		clip = 1500
		if season == 'Annual':
			latCheck = lambda x,y: x % 1 == 0 and y % 1 == 0
		rawDataPred = readNetcdf(predFilename,['lon','lat','z'],10,latCheck,notNan)

		
	#	'''
	#	Productivity boxplot
	#	'''		
		
	#	bioData = readNetcdf(bioFile,['lon','lat','z'],1,noCheck,lambda z: z<clip)
	#	biocl,bioba = plotElem(rawDataPred,bioData,noCheck,noCheck)
	#	boxplotCa(biocl,bioba,'Productivity '+season+' [mgC/m**2/day]','gmtplots/boxplots/prodBoxplot'+season+'.ps','ps',res)
	
		'''
		Nutrients boxplot
		'''

		for element in elements:
			filename = 'Nutrients/%s%s/%s%san1.001' % (element,season,elemLabel[element],seasonLabel[season])
			ncFile = filename + '.nc'
			rawData = readNetcdf(ncFile,['lon','lat','z'],1,noCheck,nonNegativity)
			Acl,Aba = plotElem(rawDataPred,rawData,noCheck,noCheck)
			boxplotCa(Acl,Aba,element + ' ['+r'$\mu$'+'mol/l]','gmtplots/boxplots/%s%s.ps' % (element,season),'ps',res)

elements 	= ['nitrate','silicate','phosphate']
elemLabel	= {'nitrate' : 'n' , 'silicate' : 'i' , 'phosphate' : 'p'}
seasons 	= ['Annual']
seasonLabel	= {'Winter' : '13', 'Summer' : '15' , 'Annual' : '00'}

files = ['seabed_lithology_ant.nc','seabed_lithology_nind.nc', 'seabed_lithology_satl.nc', 'seabed_lithology_sind.nc', 'seabed_lithology_spac.nc']#'seabed_lithology_arctic.nc', 
jointFiles = ['seabed_lithology_catl.nc', 'seabed_lithology_cpac.nc', 'seabed_lithology_natl.nc', 'seabed_lithology_npac.nc']	

def oceanPlots():

	files = ['seabed_lithology_ant.nc', 'seabed_lithology_arctic.nc', 'seabed_lithology_nind.nc', 'seabed_lithology_satl.nc', 'seabed_lithology_sind.nc', 'seabed_lithology_spac.nc']
 	jointFiles = ['seabed_lithology_catl.nc', 'seabed_lithology_cpac.nc', 'seabed_lithology_natl.nc', 'seabed_lithology_npac.nc']
	bathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',['lon','lat','z'],1,noCheck,lambda bz : bz < 0)

	for f in files:
		rawData = readNetcdf('regions/%s' % (f),['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
		rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
		boxplotCa(rcl,rba,'Bathymetry %s [m]' % (f[-7:-3]),'gmtplots/bathymetry%s.ps' % (f[-7:-3]),'ps',600)

	rawData = readNetcdf('regions/seabed_lithology_catl.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData2= readNetcdf('regions/seabed_lithology_natl.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	boxplotCa(rcl,rba,'Bathymetry %s [m]' % ('natl and catl'),'gmtplots/bathymetry%s.ps' % ('natlandcatl') ,'ps',600)

	rawData = readNetcdf('regions/seabed_lithology_cpac.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData2= readNetcdf('regions/seabed_lithology_npac.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z in [4.,5.,6.,9.,13.])
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	boxplotCa(rcl,rba,'Bathymetry %s [m]' % ('npac and cpac'),'gmtplots/bathymetry%s.ps' % ('npacandcpac') ,'ps',600)


def histOceans(binNr,cl,colrs,plotname,res):

	bathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',['lon','lat','z'],1,noCheck,lambda bz : bz < 0)
	for f in files:
		rawData = readNetcdf('regions/%s' % (f),['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
		rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
		histograms(rba,binNr,colrs,'Bathymetry %s [m]' % (f[-7:-3]),'bathymetry%s%sbins' % (f[-7:-3],binNr),res)

	rawData = readNetcdf('regions/seabed_lithology_catl.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData2= readNetcdf('regions/seabed_lithology_natl.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	histograms(rba,binNr,colrs,'Bathymetry %s [m]' % ('natl and catl'),'bathymetryNatlCatl%sbins'%(binNr),res)

	rawData = readNetcdf('regions/seabed_lithology_cpac.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData2= readNetcdf('regions/seabed_lithology_npac.nc',['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z: notNan(z) and z == cl)
	rawData['classif'].extend(rawData2['classif'])
	rawData['lon'].extend(rawData2['lon'])
	rawData['lat'].extend(rawData2['lat'])

	rcl,rba = plotElem(rawData,bathData,noCheck,noCheck)
	histograms(rba,binNr,colrs,'Bathymetry %s [m]' % ('npac and cpac'),'bathymetryNpacCpac%sbins'%(binNr),res) 



def histData(filename,cl):
	return readNetcdf(filename,['lon','lat','z'],1,lambda x,y:x%1==0 and y%1==0,lambda z:notNan(z) and z == cl)

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

def filterFun(xi,x):
	if abs(xi - x) < 1e-5: return True
	else: return False

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


def mapXY(x,y):
	return int(-89./180.*x + y)

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

	stats = DataFrame([[i+1,xvec[i],yvec[i],sali[i],temp[i],sili[i],nitr[i],phos[i],bath[i],prod[i]] for i in range(len(xvec))], index = None , columns = None)
	stats.to_csv('lithologyStats.m',sep = '\t', index = None , columns = None)

def fillList(li,rawData):

	for i,j in enumerate(rawData['classif']):

		xi = rawData['lon'][i]
		yi = rawData['lat'][i]
		ind = mapXY(xi,yi)
		li[ind] = j

	return li

def contourCa(nx,ny,nz,cont):

	'''
	Return a list with all points that the requested contour lies only
	'''

	cs = plt.contour(nx,ny,nz,[cont])
	x,y = [],[]
	for path in range(len(cs.collections[0].get_paths())):
		p = cs.collections[0].get_paths()[path]
		v = p.vertices
		x.extend(v[:,0])
		y.extend(v[:,1])
	return x,y

def readCa(filename):
	ncData = Dataset(filename,'r')
	nx = ncData.variables['lon'][:]
	ny = ncData.variables['lat'][:]
	nz = np.ma.getdata(ncData.variables['z'][:][:,:])
	return nx, ny, nz





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
	if 120 <= x <= 280 and 0 <= y <= 20:
		return 'cpac'
	if 120 <= x <= 290 and -40 <= x <= 0:
		return 'spac'
	if y > 65:
		return 'arctic'
	if y < -40:
		return 'antarctic'
	return 'none'


def histCaOceans():

	oceans = ['natl','satl','nind','sind','npac','spac','antarctic']
	conData = readNetcdf('test.nc',['lon','lat','z'],1,noCheck,notNan)

	for o in oceans:
		if o == 'natl':
			oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'catl'
		if o == 'npac':
			oceanCheck = lambda x,y: ocean(x,y) == o or ocean(x,y) == 'cpac'
		else:
			oceanCheck = lambda x,y: ocean(x,y) == o
		bathData = readNetcdf('gmtplots/netcdffiles/bathymetryMasked.nc',['lon','lat','z'],1,oceanCheck,notNan)
		ccl,cba = plotElem(conData,bathData,noCheck,noCheck)
		if cba:
			histograms(cba,20,colors[4],'Bathymetry at CaCO3 20 %s' % (o),'caco3bath%s' % (o) ,600)









































