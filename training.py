import re
from sklearn.ensamble import RandomForestClassifier

def readTxt(filename):

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

#rawData is what readTxt returns, data is the data on the format that the ML functions want
def training(rawData):

		data = [[rawData['lon'][i] , rawData['lat'][i]] for i in range(len(rawData['lon']))]
		labels = rawData['classif']
		rfc = RandomForestClassifier(n_estimators=100)
		rfc.fit(data, labels)

'''
Taken from http://stackoverflow.com/questions/13796315/plot-only-on-continent-in-matplotlib
had to change the code to adapt to updates in matplotlib
'''

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
        result.extend(points[mask])
        points = points[~mask]
    return np.array(result)

points = np.random.randint(0, 90, size=(100000, 2))
m = Basemap(projection='moll',lon_0=0,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
x, y = m(points[:,0], points[:,1])
loc = np.c_[x, y]
polys = [p.boundary for p in m.landpolygons]
land_loc = points_in_polys(loc, polys)
m.plot(land_loc[:, 0], land_loc[:, 1],'ro')
plt.show()