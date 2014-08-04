import xlrd

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