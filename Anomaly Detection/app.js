var request = require('request');
config = require('./conf.js');

//var first_arg = process.argv[2];
//var d = load('./Data/test2.txt');

var fs = require('fs');
const article = fs.readFileSync("./Data/rawData/data.txt");
lineArray = article.toString().split('\n');
const result = fs.readFileSync("./Data/result/anomaly_detection_result.txt");
lineArray2 = result.toString().split('\n');
//fs.readFile("./Data/test2.txt", 'utf8', function(err, d){

//});

var data = {
		"m2m:cin": {
			//"or": lineArray2,
			"con": lineArray2 // + "," + lineArray //first_arg
		}
};
var headers = {
	 'cache-control': 'no-cache',
	'Content-Type': 
	'application/vnd.onem2m-res+json; ty=4',
	'X-M2M-Origin': config.ae.id,
	'X-M2M-RI': config.ae.name,
	Accept: 'application/json' 
};
var useprotocol = 'http';
var uri = useprotocol
+'://'+config.cse.host+':'+config.cse.port+'/'+config.cse.name+'/'+config.ae.name+'/Gateway';
console.log(uri);
var options = {
	method: 'POST',
	url: uri,
	headers:headers,
	body: JSON.stringify(data)
};
request(options, function (error, response, body) {
	if (error) throw new Error(error);
	console.log(body);
});
