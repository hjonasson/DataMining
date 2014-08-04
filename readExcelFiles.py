import xlrd

#I try to keep the design open, however some functions have to be problem specific, they will come second in the file

def reader(filename,ind = 0):

	#Reads in a file, for example 'Deck41_classified_final.xlsx'
	#returns sheet number ind, the first sheet if unspecified
	
	workbook = xlrd.open_workbook(filename)
	return workbook.sheet_by_index(ind)

def createDictionary(sheet):

	#Creates a dictionary to work with
	#the first value of each column is the keys
	
	data = {}
	metrics = lambda i : sheet.cell_value(0,i)
	values = lambda i: [sheet.cell_value(j,i) for j in range(1,sheet.nrows)]

	for i in range(sheet.ncols):
		data[metrics(i)] = values(i)

	return data

def testing():
	assert type(reader('Deck41_classified_final.xlsx')) == xlrd.sheet.Sheet
	sheet = reader('Deck41_classified_final.xlsx')
	data = createDictionary(sheet)
	assert sheet.cell_value(0,0) in data
	assert sheet.ncols == len(data)
	print 'Tests passed'

#Problem specific function
#
#To do
#
#Find all points classified as mud
#Make a window around it
#Find all points in the window
#Define criteria for changing mud classification
#If criteria met, change mud classification

#def findMudPoints()
#	gives a list of mud points
#def pointsInBox()
#	finds all points in a box around mud point i
#def criteria()
#	assesses whether the points in box allow mud point to be reclassified
#def rewrite()
#	makes a new file to me handed to the gmt