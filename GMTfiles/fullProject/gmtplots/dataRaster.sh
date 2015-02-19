#!/bin/zsh
# gmt gmtset MAP_GRID_PEN_PRIMARY 0.25p

gmt gmtset GMT_INTERPOLANT none FONT_LABEL 14p MAP_ANNOT_OFFSET_PRIMARY 0.3 MAP_LABEL_OFFSET 0.2 MAP_FRAME_TYPE fancy

version=v6

# Mercator projection
region=-180/180/-75/75
proj=M26

# dummy variable to call min and max for the .cpt files
j=0
for i in $( ls netcdffiles ); do

	grid=netcdffiles/$i

	# ps file
	psfile=psfiles/${i:0:${#i}-3}.ps

	# color file
	color=cptfiles/${i:0:${#i}-3}.cpt

	# perform histogram equalization
	
	# if you dont want to perform
	gridhisteq=$grid
	#gridhisteq=histeqfiles/${i:0:${#i}-3}histeq.nc
	#gmt grdhisteq $grid -G$gridhisteq

	# make a color file .cpt
	#gmt grd2cpt $gridhisteq -L > $color

	# make a raster image
	gmt grdimage -J$proj -R$region $gridhisteq -C$color -K > $psfile

	# outlines
	gmt psxy PB2002_boundaries.xy -R -J$proj -O -K  >> $psfile
	gmt psxy LIPS.xy -R -J$proj -O -K >> $psfile
	gmt psxy separate.xy -Sc0.15i -W2 -R -J$proj -O -K >> $psfile
	gmt pscoast -R -J$proj -B30g10/10g10WeSn -A0/1/1 -Dl -G220 -I1 -W1 -O  >> $psfile

	# make into a pdf
	gmt ps2raster -Tg $psfile -E300 -A
	open $psfile
	j+=1
;
done