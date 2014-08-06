import re
import xlrd


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
#Filter classification (mud)
#Use filtered classification and make boxes around
#Find all points in the box
#Assess the points in the box
#Rewrite the file

def filterClassification(data,filter):

	filteredData = {'lon':[],'lat':[],'classif':[],'ind':[]}
	for i in range(len(data['lon'])):
		if data['classif'][i] == filter:
			filteredData['lon'].append(data['lon'][i])
			filteredData['lat'].append(data['lat'][i])
			filteredData['classif'].append(data['classif'][i])
			filteredData['ind'].append(data['ind'][i])
	return filteredData


#reader and createDictionary handle the xlsx files that I need to work with
def reader(filename,ind = 0):

	#Reads in a file, for example 'Deck41_classified_final.xlsx'
	#returns sheet number ind, the first sheet if unspecified
	
	workbook = xlrd.open_workbook(filename)
	return workbook.sheet_by_index(ind)

def createDictionary(sheet):

	#Creates a dictionary to work with
	#the first value of each column from the sheet gives the keys
	
	data = {}
	metrics = lambda i : sheet.cell_value(0,i)
	values = lambda i: [sheet.cell_value(j,i) for j in range(1,sheet.nrows)]

	for i in range(sheet.ncols):
		data[metrics(i)] = values(i)

	return data


#listByClassification can take in the dictionary from createDictionary, variable longitude or latitude etc. classifier is here 'Classification' (mud, ooze etc.), classifValue is 9 for mud
def listByClassification(data,variable,classifier = 'Classification',classifValue = 9.0):
	return [data[variable][i] for i in range(len(data[variable])) if data[classifier][i] == classifValue]


def findMudPoints(data):
	return {'lon':listByClassification(data,'Longitude'),'lat':listByClassification(data,'Latitude')}

#Finds all points in a box of size 2dx*2dy around a given point, hierarcy: pointsInBox -> BoxedPoints
def pointsInBox(centerPoint,data,dx,dy):
	xmin = centerPoint[0] - dx
	xmax = centerPoint[0] + dx
	ymin = centerPoint[1] - dy
	ymax = centerPoint[1] + dy
	boxed = boxedPoints(data,xmin,xmax,ymin,ymax)
	return boxed

def boxedPoints(data,xmin,xmax,ymin,ymax):
	
	boxed = {'lon':[],'lat':[],'classif':[],'ind':[]}
	for i in range(len(data['Longitude'])):
		x = data['Longitude'][i]
		y = data['Latitude'][i]
		if x > xmin:
			if x < xmax:
				if y > ymin:
					if y < ymax:
						boxed['lon'].append(x)
						boxed['lat'].append(y)
						boxed['classif'].append(data['Classification'][i])
						boxed['ind'].append(i)						
	return boxed

def testing():
	assert type(reader('Deck41_classified_final.xlsx')) == xlrd.sheet.Sheet
	sheet = reader('Deck41_classified_final.xlsx')
	data = createDictionary(sheet)
	assert sheet.cell_value(0,0) in data
	assert sheet.ncols == len(data)
	assert len(listByClassification(data,'Longitude')) 	#There should me some measurements classified as mud
	assert len(findMudPoints(data)) == 2 				#For latitude and longitude
	print 'Tests passed'

#To do
#
#Find all points classified as mud
#Make a window around it
#Find all points in the window
#Define criteria for changing mud classification
#If criteria met, change mud classification


#def criteria()
#	assesses whether the points in box allow mud point to be reclassified
#def rewrite()
#	makes a new file to me handed to the gmt