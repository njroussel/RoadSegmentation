var page = require("webpage").create(),
	url = "http://localhost:63342/TileDownloader/tmp.html";

page.viewportSize = {
	  width: 800,
	  height: 800
};

function onPageReady() {
	page.clipRect = {
		top: 0,
		left: 0,
		width: 400,
		height: 400
	}
	page.render('meteo.png');
	console.log('Capture saved');
	phantom.exit();
}

page.open(url, function (status) {
	function checkReadyState() {
		setTimeout(function () {
			var readyState = page.evaluate(function () {
				return document.readyState;
			});

			if ("complete" === readyState) {
				onPageReady();
			} else {
				checkReadyState();
			}
		});
	}

	checkReadyState();
});
