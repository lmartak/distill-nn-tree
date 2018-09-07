#!/bin/bash

IMDIR="$(pwd)/assets/img"
name="$1" # expecting name of the folder with image frames
srcdir="$IMDIR/$name"
dstdir="$srcdir"num

if [ ! -d "$srcdir" ]; then
	echo "$srcdir is not a directory"
	exit 1
fi

# create temporary dir to store labeled frames in
mkdir -p $dstdir

# generate labeled frames in temporary dir
n_max=`ls $srcdir | cut -d "." -f 1 | sort -h | tail -n 1 | sed 's/^0*//'`
for fn in `ls $srcdir `; do
	n=`echo $fn | cut -d "." -f 1 | sed 's/^0*//'`;
	convert -pointsize 20 -fill blue \
		-draw "text 150,50 '"$name"s passed:'" \
		-draw "text 150,75 '$n / $n_max'" \
		$srcdir/$fn $dstdir/$fn;
done

# generate gif from labeled frames
convert -delay 10 -loop 0 $dstdir/* $srcdir.gif

# remove temporary dir with all its contents
rm -r $dstdir

exit 0
