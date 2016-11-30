var roadPage = require("webpage").create();
const ROAD_URL = "http://localhost:63342/TileDownloader/tmp.html";
var satellitePage = require("webpage").create();
const SAT_URL = "http://localhost:63342/TileDownloader/tmp.html";

var width = 1000;
var height = 2000;

const FOLDER_ROAD = 'groundtruth';
const FOLDER_SATELLITE = 'images';
const TILE_SIZE = 400;
const TILE_NAME = 'satImage';
const EXTENSION = '.png';

roadPage.viewportSize = {
  width: width + TILE_SIZE,
  height: height + TILE_SIZE
};

fetchTilesOnPageReady(roadPage, ROAD_URL, FOLDER_ROAD);
fetchTilesOnPageReady(satellitePage, SAT, FOLDER_SATELLITE);


function fetchTiles(page, folder) {
  const heightTilesCount = Math.floor(height / TILE_SIZE);
  const widthTilesCount = Math.floor(width / TILE_SIZE);

  for (var i = 0; i < widthTilesCount; i++) {
    for (var j = 0; j < heightTilesCount; j++) {
      page.clipRect = {
        top: i * TILE_SIZE,
        left: j * TILE_SIZE,
        width: TILE_SIZE,
        height: TILE_SIZE
      };
      page.render(folder + '/' + TILE_NAME + (i + j) + EXTENSION);
    }
  }
}

function fetchTilesOnPageReady(page, pageUrl, folder){
  page.open(pageUrl, function (status) {
    function checkReadyState() {
      setTimeout(function () {
        var readyState = page.evaluate(function () {
          return document.readyState;
        });

        if ("complete" === readyState) {
          fetchTiles(page, folder);
          phantom.exit();
        } else {
          checkReadyState();
        }
      });
    }
    checkReadyState();
  });
}
