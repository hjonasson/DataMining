import os

'''
This file holds functions that take mapped data and write them to files
Also for writing postscript files that call gmt
'''

def makeMap(dx,withData,projection = 'M',newFilename = 'seabed_raster.sh'):
	
	'''
	makeMap writes a postscript file that calls GMT to view the mapped data
	'''
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
	g.write('infile1=seabed_lithology_blocks_regular_grid.txt\n')
	g.write('scalepar="-D0/-0.4/10.0/1"\n')
	g.write('\n')
	g.write('gmt xyz2grd -Rd $infile1 -Gtest.nc -I'+str(dx)+'d -V\n')
	g.write('gmt grdimage -R$region -J'+str(projection)+'26 test.nc -C$color -V -X2.5 -Y2 -K > $psfile\n')
	g.write('gmt pscoast -R$region -J'+str(projection)+'26 -B30g10/10g10WeSn -Di -I1 -W1 -G220 -O -K >> $psfile\n')
	g.write('gmt psxy PB2002_boundaries.xy -R -J -W2 -m -O -K >> $psfile\n')
	if withData:
		g.write('gmt psxy fullData.txt -R$region -J'+str(projection)+'26 -C$color -Sc0.085 -O -K >> $psfile\n')
	g.write('gmt psxy LIPS.xy -R -J -W2 -m -O >> $psfile\n')
	g.write('gmt ps2raster -Tj $psfile -A -V \n')
	g.write('\n')
	g.write('open $psfile\n')
	g.close()


def gmtMap(filename='seabed_raster.sh',makeExecutable = True):

	'''
	gmtMap calls the file created in makeMap
	'''

	if makeExecutable:
		os.system('chmod +x '+filename)
	os.system('./'+filename)		

def writePredictions(xpoints,ypoints,prediction,header = '> Predictions made with training.py',newFilename = 'seabed_lithology_regular_grid.txt'):

	'''
	writePredictions writes the predictions made in mapping.py to a three column .txt file
	'''
	g = open(newFilename,'w')
	g.write(header)
	g.write('\n')
	for i in range(len(xpoints)):
		xi = xpoints[i]
		for j in range(len(ypoints)):
			yi = ypoints[j]
			prei = prediction[i][j]
			line = str(xi) + ' ' + str(yi) + ' ' + str(prei) + '\n'
			g.write(line)
	g.close 
