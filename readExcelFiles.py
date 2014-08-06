import re

#readTxt reads in the three column data files, the headers start with >, so they are filtered out by that
#All line numbers are kept in order so later the files can be altered in an easy manner
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

#Todo
#
#Use filtered classification and make boxes around
#Find all points in the box
#Assess the points in the box
#Rewrite the file

def filterClassification(data,filt=9):

	filteredData = {'lon':[],'lat':[],'classif':[],'ind':[]}
	for i in range(len(data['lon'])):
		if data['classif'][i] == filter:
			filteredData['lon'].append(data['lon'][i])
			filteredData['lat'].append(data['lat'][i])
			filteredData['classif'].append(data['classif'][i])
			filteredData['ind'].append(data['ind'][i])
	return filteredData

def classifyPoints(data,dx,dy,filt=9):
	
	filteredData = filterClassification(data,filt)
	for i in range(len(filteredData['lon'])):
		x = data['lon'][i]
		y = data['lat'][i]
		centerPoint = (x,y)
		box = pointsInBox(centerPoint,data,dx,dy)
		#call classifyBox

	#return a list of points that need to be changed in the file
	#give this to a function that makes an altered file
	return True













def classifyBox(box):
	#find out what to do here
	return True

#Finds all points in a box of size 2dx*2dy around a given point, hierarcy: pointsInBox -> BoxedPoints
def pointsInBox(centerPoint,data,dx,dy):
	xmin = centerPoint[0] - dx
	xmax = centerPoint[0] + dx
	ymin = centerPoint[1] - dy
	ymax = centerPoint[1] + dy
	boxed = boxedPoints(data,xmin,xmax,ymin,ymax)
	return boxed

#boxedPoints gives a dictionary of all points within a certain box on the map
def boxedPoints(data,xmin,xmax,ymin,ymax):
	
	boxed = {'lon':[],'lat':[],'classif':[],'ind':[]}
	for i in range(len(data['lon'])):
		x = data['lon'][i]
		y = data['lat'][i]
		if x > xmin:
			if x < xmax:
				if y > ymin:
					if y < ymax:
						boxed['lon'].append(x)
						boxed['lat'].append(y)
						boxed['classif'].append(data['classif'][i])
						boxed['ind'].append(i)						
	return boxed

