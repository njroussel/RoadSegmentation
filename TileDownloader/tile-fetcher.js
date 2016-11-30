var webPage = require("webpage");
var system = require("system");

var page = webPage.create();
const url = system.args[1];
const folder = system.args[2];

var width = 8000;
var height = 8000;

const TILE_SIZE = 400;
const TILE_NAME = 'satImage';
const EXTENSION = '.jpg';

page.viewportSize = {
  width: width + TILE_SIZE,
  height: height + 2 * TILE_SIZE
};

openPageAndFetchTiles(page, url, folder);

function fetchTiles(page, folder) {
  const heightTilesCount = Math.floor(height / TILE_SIZE);
  const widthTilesCount = Math.floor(width / TILE_SIZE);

  for (var i = 0; i < widthTilesCount; i++) {
    for (var j = 0; j < heightTilesCount; j++) {
      page.clipRect = {
        top: height - (i * TILE_SIZE),
        left: j * TILE_SIZE,
        width: TILE_SIZE,
        height: TILE_SIZE
      };

      page.render(folder + '/' + TILE_NAME + (i * widthTilesCount + j + 101) + EXTENSION, {format: 'jpeg', quality: '100'});
      console.log("Fetched tile " + (i * widthTilesCount + j) + "/" + (widthTilesCount * heightTilesCount - 1))
    }
  }

  phantom.exit();
}

function openPageAndFetchTiles(page, url, folder) {
  page.open(url, function (status) {
    function checkReadyState() {
      setTimeout(function () {
        var readyState = page.evaluate(function () {
          return document.readyState;
        });

        if ("complete" === readyState) {
            fetchTiles(page, folder);
        } else {
          checkReadyState();
        }
      });
    }

    checkReadyState();
  });
}