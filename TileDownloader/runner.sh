#!/usr/bin/env bash

phantomjs tile-fetcher.js road.html groundtruth
echo "Road tiles finished !"
phantomjs tile-fetcher.js satellite.html images
echo "Satellite tiles finished !"
