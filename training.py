import re
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import sklearn
import readFullData
from mpl_toolkits.basemap import Basemap


'''
To do

Check statistics
Make map

'''

def testing():

	# Test if the test file is in the directory
	filename = 'GMTfiles/seabed_lithology_v4.txt'
	assert os.path.exists(filename)

	# Test if there is any data in the file
	rawData = readTxt(filename)
	assert rawData['lon']

	# Test if random forest classifier is not empty
	rfc = training(rawData)
	assert type(rfc[0]) == sklearn.tree.tree.DecisionTreeClassifier

	# Test if there are points left after cleaning put continents
	xpoints = np.arange(-180,180,90)
	ypoints = np.arange(-90,90,45)
	m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
	seaMap = cleanContinents(m,xpoints,xpoints)
	for i in seaMap: assert len(ypoints) >= len(i)

	# Test if predictions were made on each point of the map
	prediction = mapping(xpoints,seaMap,rfc)
	for i in range(len(prediction)): assert len(prediction[i]) == len(seaMap[i])

	#Tests have passed
	print 'All tests passed'

#This is a file that produces the correct raster images and will serve as a prototype for automation

def makeMap(dx,dataFile = 'seabed_lithology_regular_grid.txt',newFilename = 'seabed_raster.sh'):
	g = open(newFilename,'w')
	print 'Writing postscript file'
	g.write('#!/bin/zsh\n')
	g.write('gmt gmtset GRID_PEN_PRIMARY 0.25p\n')
	g.write('\n')
	g.write('version=2\n')
	g.write('\n')
	g.write('region=-180/180/-75/75\n')
	g.write('color=seabed_lithology.cpt\n')
	g.write('psfile=seabed_lithology_reviewed_all_v$version.ps\n')
	g.write('infile1='+dataFile+'\n')
	g.write('scalepar="-D0/-0.4/10.0/1"\n')
	g.write('\n')
	g.write('gmt xyz2grd -Rd $infile1 -Gtest.nc -I'+str(dx)+'d -V\n')
	g.write('gmt grdimage -R$region -JM26 test.nc -C$color -V -X2.5 -Y2 -K > $psfile\n')
	g.write('gmt pscoast -R$region -JM26 -B30g10/10g10WeSn -Di -I1 -W1 -G220 -O -K >> $psfile\n')
	g.write('gmt psxy PB2002_boundaries.xy -R -J -W2 -m -O -K >> $psfile\n')
	g.write('gmt psxy LIPS.xy -R -J -W2 -m -O >> $psfile\n')
	g.write('gmt ps2raster -Tj $psfile -A -V \n')
	g.write('\n')
	g.write('open $psfile\n')
	g.close()



def running(dx,dy,filename = 'GMTfiles/seabed_lithology_v4.txt'):

	xpoints = np.arange(-180,180,dx)
	ypoints = np.arange(-90+dy,90,dy)
	#rawData = readTxt(filename)
	rawData = readFullData.readCols(filename)
	rfc = training(rawData)
	m = Basemap(projection='merc',llcrnrlat=-90,urcrnrlat=90,\
				llcrnrlon=-180,urcrnrlon=180,lat_ts=5,resolution='c')#res c,l,i,h,f, has big effects on efficiency
	seaMap = cleanContinents(m,xpoints,ypoints)
	prediction = mapping(xpoints,seaMap,rfc)
	writePredictions(xpoints,seaMap,prediction, header = '> Predictions made with training.py',newFilename = 'seabed_lithology_regular_grid.txt')
	makeMap(dx)
	gmtMap()

def gmtMap(filename='seabed_raster.sh',makeExecutable = True):

	if makeExecutable:
		os.system('chmod +x '+filename)
	os.system('./'+filename)		

def writePredictions(xpoints,seaMap,prediction,header = '> Predictions made with training.py',newFilename = 'seabed_lithology_regular_grid.txt'):

	g = open(newFilename,'w')
	newline = '\n'
	g.write(header)
	g.write(newline)
	for i in range(len(xpoints)):
		xi = xpoints[i]
		for j in range(len(seaMap[i])):
			yi = seaMap[i][j]
			prei = prediction[i][j][0]
			line = str(xi) + ' ' + str(yi) + ' ' + str(prei) +newline
			g.write(line)
	g.close 




def cleanContinents(m,xpoints,ypoints):

	'''
	Takes in the projection m and the grid of x and y
	clears out the points that are inside continents

	returns a matrix where each column corresponds to a value in xpoints
	'''

	seaMap = [[j for j in ypoints if not Basemap.is_land(m,m(i,j)[0],m(i,j)[1])] for i in xpoints]

	return seaMap

def mapping(xpoints,seaMap,rfc):

	'''
	Takes in a grid and a classifier that has already been trained
	uses the classifier to predict values on the grid

	the grid is in vectors (x=[-180...180] for example)
	'''
	prediction = []
	i = 1.
	j = 0
	for xi in xpoints:
		xpredict = []
		for yi in seaMap[j]:
			xpredict.append(rfc.predict([xi,yi]))
		prediction.append(xpredict)
		print 'Finished '+str(i/len(xpoints))+' %'
		i += 1
		j += 1

	return prediction

def readTxt(filename):

	'''
	Takes in a filename of a three column format and gives
	the data from it in a dictionary

	seabed_lithology_v4.txt for example
	'''

	data = {'lon':[],'lat':[],'classif':[],'ind':[]}
	f = open(filename)
	lines = f.readlines()
	lineNr = 0
	for line in lines:
		if line[0] != '>':
			splitLine = re.split(r'\t+',line.strip())
			if len(splitLine) == 3:
				data['lon'].append(float(splitLine[0]))
				data['lat'].append(float(splitLine[1]))
				data['classif'].append(int(splitLine[2]))
				data['ind'].append(lineNr)
		lineNr += 1
	return data

def training(rawData):

	'''
	Takes in the data from readTxt and sorts it to the format ML
	package wants, then trains a classifier
	'''

	data = [[rawData['lon'][i] , rawData['lat'][i]] for i in range(len(rawData['lon']))]
	labels = rawData['classif']
	rfc = RandomForestClassifier(n_estimators=100)
	rfc.fit(data, labels)
	return rfc





















'''
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c') #resolution can be c,l,i,h,f

#Klippa fyrst ut punkta a landi til thess ad spara tima
xpoints = np.arange(-180,180,5)
ypoints = np.arange(-90,90,5)
x, y = m(xpoints, ypoints)

'''


'''
Taken from http://stackoverflow.com/questions/13796315/plot-only-on-continent-in-matplotlib
had to change the code to adapt to updates in matplotlib


from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib

def points_in_polys(points, polys):
    result = []
    for poly in polys:
    	verts = []
    	codes = []
    	for point in poly:
    		verts.append((point[0],point[1]))
    		codes.append(Path.LINETO)
    	codes[0] = Path.MOVETO
    	codes[-1] = Path.CLOSEPOLY
    	path = Path(verts,codes)
        mask = matplotlib.path.Path.contains_points(path,points)
        #nonmask = [not i for i in mask]
        #result.extend(points[[i for i in range(len(points)) if i not in mask]])
        result.extend(points[mask])
        points = points[~mask]
    return np.array(result)

points = np.random.randint(0, 90, size=(100000, 2))
m = Basemap(projection='merc',lon_0=0,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
x, y = m(points[:,0], points[:,1])
loc = np.c_[x, y]
polys = [p.boundary for p in m.landpolygons]
land_loc = points_in_polys(loc, polys)
m.plot(land_loc[:, 0], land_loc[:, 1],'ro')
plt.show()
'''