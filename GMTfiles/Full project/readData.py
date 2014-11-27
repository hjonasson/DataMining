import re
import xlrd

'''
This file has functions that are intended to read all forms of data for the project of mapping lithology to a a regular grid
'''


def readCols(workbookName,cols=[6,7,14],labels=['lat','lon','classif']):

	'''
	Takes in an excel file and reads in the necessary information.
	Gives it back in a dictionary

	{'lon':[1,2,3],'lat':[1,2,3],'classif':[1,1,1]} for example of taking the longitude, latitude and corresponding classification
	'''

	workbook = xlrd.open_workbook(workbookName)
	worksheet = workbook.sheet_by_index(0)
	data = {labels[i]:worksheet.col_values(cols[i])[2:-1] for i in range(len(labels))}

	return data

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
