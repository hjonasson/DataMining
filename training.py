import re
from sklearn.ensamble import RandomForestClassifier
import numpy as np

'''
To do

Write tests
Clean out points within continents


'''

def testing():
	print 'All tests passed'

def mapping(xpoints,ypoints,rfc):

'''
Takes in a grid and a classifier that has already been trained
uses the classifier to predict values on the grid

the grid is in vectors (x=[-180...180] for example)
'''

	prediction = []
	i = 0.
	for xi in xpoints:
		xpredict = []
		for yi in ypoints:
			xpredict.append(rfc.predict([xi,yi]))
		prediction.append(xpredict)
		print 'Finished '+str(i/len(xpoints))+' %'
		i += 1

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