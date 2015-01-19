from collections import Counter
import numpy as np
import xlrd
import xlwt

'''
This file has functions that are intended to read excel files and find duplicates
'''

def writeDuplicates(workbookName):

	'''
	Reading
	'''
	workbook = xlrd.open_workbook(workbookName)
	sheet = workbook.sheet_by_index(0)
	data = [[sheet.cell_value(row, col) for col in range(sheet.ncols)] for row in range(sheet.nrows)]

	'''
	Processing
	'''
	indices = {}
	for i in range(2,len(data)):
		indices.setdefault((data[i][6],data[i][7]), []).append(i)

	'''
	Writing
	'''
	book = xlwt.Workbook()
	sheet = book.add_sheet('duplicate_data')

	allMultis = []
	for i in indices:
		if len(indices[i]) == 1:
			#correspData = [data[j][14] for j in indices[i]]
			#if len(set(correspData)) > 1:
			allMultis.extend(indices[i])

	newData = [data[j] for j in allMultis]

	for row, c in enumerate(newData):
		for col, d in enumerate(c):
			sheet.write(row,col,d) 

	book.save('Ocean_sediments_no_Doubles.xls')




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
