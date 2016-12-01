#!/usr/bin/env bash

echo "Fetching road tiles from Google Maps"
phantomjs tile-fetcher.js road.html groundtruth $1
echo "Road tiles finished !"
echo ""
echo ""

echo "Fetching satellite tiles from Google Maps"
phantomjs tile-fetcher.js satellite.html images $1
echo "Satellite tiles finished !"
echo ""
echo ""


