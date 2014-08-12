from collections import Counter
import re

#readTxt reads in the three column data files, the headers start with >, so they are filtered out by that
#All line numbers are kept in order so later the files can be altered in an easy manner

dataFile = 'GMTfiles/seabed_lithology_v4.txt'
dx = 10
dy = 10
minNr = 10

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
"""
f = open(dataFile)
lines = f.readlines()
for ind in changes['ind']:
	splitLine = re.split(r'\t+',lines[+1].strip())
	print splitLine[2]
"""

def testing(dataFile = 'GMTfiles/seabed_lithology_v4.txt',dx = 10,dy = 10, minNr = 10,filt = 9):
	data = readTxt(dataFile)
	filteredData = filterClassification(data)
	assert all(filteredData['classif'][0] == item for item in filteredData['classif']) #All filtered items have the same classification
	changes = classifyPoints(data,dx,dy,minNr)
	assert all(filt == data['classif'][i] for i in changes['ind'])

def filterClassification(data,filt=9):

	filteredData = {'lon':[],'lat':[],'classif':[],'ind':[]}
	for i in range(len(data['lon'])):
		if data['classif'][i] == filt:
			filteredData['lon'].append(data['lon'][i])
			filteredData['lat'].append(data['lat'][i])
			filteredData['classif'].append(data['classif'][i])
			filteredData['ind'].append(i)
	return filteredData

#classifyPoints returns the points that need to be changed in a file and their lineNr
def classifyPoints(data,dx,dy,minNr,filt=9):
	
	filteredData = filterClassification(data,filt)
	changes = {'ind':[],'newClass':[]}
	for i in filteredData['ind']:
		x = data['lon'][i]
		y = data['lat'][i]
		centerPoint = (x,y)
		box = pointsInBox(centerPoint,data,dx,dy)
		classify = classifyBox(box,minNr)
		if classify != False:
			#add to a list of changes to be made
			changes['ind'].append(i)
			changes['newClass'].append(classify)
	print 'Changes to be made are '+str(len(changes['ind']))
	return changes

def rewrite(filename,changes):

	f = open(filename)
	lines = f.readlines()
	count = 0
	for i in changes['ind']:
		splitLine = re.split(r'\t+',lines[i].strip())
		splitLine[2] = str(changes['newClass'][count])
		lines[i] = splitLine[0] + '\t' + splitLine[1] + '\t' + splitLine[2] + '\n'
		count += 1
	newFileName = filename[:-4] + 'new' + '.txt'
	g = open(newFileName,'w')
	for line in lines:
		g.write(line)
	g.close 

#As for now, the classification is such that if there are minNr of points of a certain classification in the box, then the point is changed to that
def classifyBox(box, minNr):

	filt = 9
	classif = [i for i in box['classif'] if i != filt]
	try:	
		mostCommon = max(set(classif),key=classif.count)
		nrOfMostCommon = classif.count(mostCommon)
		return mostCommon
	except:
		return False

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

data = readTxt(dataFile)
changes2 = classifyPoints(data,dx,dy,minNr)
rewrite(dataFile,changes)
