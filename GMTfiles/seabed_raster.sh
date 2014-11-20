#This is a file that produces the correct raster images and will serve as a prototype for automation

#!/bin/zsh
cp ../seabed_lithology_regular_grid.txt .
gmt gmtset GRID_PEN_PRIMARY 0.25p

version=2

region=-180/180/-75/75
color=seabed_lithology.cpt
psfile=seabed_lithology_reviewed_all_v$version.ps
infile1=seabed_lithology_regular_grid.txt
scalepar="-D0/-0.4/10.0/1"

gmt xyz2grd -Rd $infile1 -Gtest.nc -I2d -V
gmt grdimage -R$region -JM26 test.nc -C$color -V -X2.5 -Y2 -K > $psfile
gmt pscoast -R$region -JM26 -B30g10/10g10WeSn -Di -I1 -W1 -G220 -O -K >> $psfile
gmt psxy PB2002_boundaries.xy -R -J -W2 -m -O -K >> $psfile
gmt psxy LIPS.xy -R -J -W2 -m -O >> $psfile

gmt ps2raster -Tj $psfile -A -V 

open $psfile
