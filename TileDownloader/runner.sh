#!/usr/bin/env bash

echo "Fetching road tiles from Google Maps"
phantomjs tile-fetcher.js road.html groundtruth
echo "Road tiles finished !"
echo ""
echo ""

echo "Fetching satellite tiles from Google Maps"
phantomjs tile-fetcher.js satellite.html images
echo "Satellite tiles finished !"
echo ""
echo ""

echo "Binarizing road tiles"
for file in `ls ./groundtruth/satImage*.jpg`; do
	convert $file -fuzz 1% -fill 'rgb(0,0,0)' -opaque 'rgb(228, 227,223)' $file
	convert $file -fuzz 50% -fill 'rgb(255, 255, 255)' -opaque 'rgb(255, 255, 255)' $file
done

echo "Success !"
