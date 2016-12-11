#!/usr/bin/env bash

sed -i '156s/.*/      center: new google.maps.LatLng(37.123617, -120.263638),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(37.123617, -120.263638),/' satellite.html
. fetchTiles.sh 101


sed -i '156s/.*/      center: new google.maps.LatLng(36.322263, -119.319836),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(36.322263, -119.319836),/' satellite.html
. fetchTiles.sh 201


sed -i '156s/.*/      center: new google.maps.LatLng(36.804362, -119.769992),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(36.804362, -119.769992),/' satellite.html
. fetchTiles.sh 301


sed -i '156s/.*/      center: new google.maps.LatLng(37.754000, -122.432041),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(37.754000, -122.432041),/' satellite.html
. fetchTiles.sh 401


echo "Binarizing road tiles"
for file in `ls ./groundtruth/satImage*.png`; do
	convert $file -fuzz 1% -fill 'rgb(0,0,0)' -opaque 'rgb(228, 227,223)' $file
	convert $file -fuzz 50% -fill 'rgb(255, 255, 255)' -opaque 'rgb(255, 255, 255)' $file
done

echo "Success !"