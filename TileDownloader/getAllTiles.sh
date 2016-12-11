#!/usr/bin/env bash

#Chowchilla
sed -i '156s/.*/      center: new google.maps.LatLng(37.123617, -120.263638),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(37.123617, -120.263638),/' satellite.html
. fetchTiles.sh 101

#Visalia
sed -i '156s/.*/      center: new google.maps.LatLng(36.322263, -119.319836),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(36.322263, -119.319836),/' satellite.html
. fetchTiles.sh 201

#Fresno
sed -i '156s/.*/      center: new google.maps.LatLng(36.804362, -119.769992),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(36.804362, -119.769992),/' satellite.html
. fetchTiles.sh 301

#San Francisco
sed -i '156s/.*/      center: new google.maps.LatLng(37.754000, -122.432041),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(37.754000, -122.432041),/' satellite.html
. fetchTiles.sh 401

#San Jose
sed -i '156s/.*/      center: new google.maps.LatLng(37.263494, -121.906921),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(37.263494, -121.906921),/' satellite.html
. fetchTiles.sh 501

#Santa Barbara
sed -i '156s/.*/      center: new google.maps.LatLng(34.429336, -119.709984),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(34.429336, -119.709984),/' satellite.html
. fetchTiles.sh 601

#Kansas City
sed -i '156s/.*/      center: new google.maps.LatLng(39.023648, -94.576525),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(39.023648, -94.576525),/' satellite.html
. fetchTiles.sh 701

#Detroit
sed -i '156s/.*/      center: new google.maps.LatLng(42.369367, -83.201832),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(42.369367, -83.201832),/' satellite.html
. fetchTiles.sh 801

#Philadelphie
sed -i '156s/.*/      center: new google.maps.LatLng(39.998334, -75.140165),/' road.html
sed -i '32s/.*/      center: new google.maps.LatLng(39.998334, -75.140165),/' satellite.html
. fetchTiles.sh 901


echo "Binarizing road tiles"
for file in `ls ./groundtruth/satImage*.png`; do
	convert $file -fuzz 1% -fill 'rgb(0,0,0)' -opaque 'rgb(228, 227,223)' $file
	convert $file -fuzz 50% -fill 'rgb(255, 255, 255)' -opaque 'rgb(255, 255, 255)' $file
done

echo "Success !"