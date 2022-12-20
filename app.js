const fs = require('fs');
const markdown = require('markdown-it')();
const express = require("express");
const path = require("path");
const favicon = require('serve-favicon');
const logger = require("morgan");
const compress = require('compression');
const cookieParser = require("cookie-parser");
const bodyParser = require("body-parser");
const ejs = require('ejs').__express;
const helmet = require('helmet');

const { execFile } = require('child_process');
const tmp = require('tmp');

let app = express();

app.set('port', process.env.PORT || 6000);
app.set("views", path.join(__dirname, "views"));
app.set('view engine', 'ejs');
app.engine('ejs', ejs);
app.engine('.html', ejs);

app.engine('md', function (path, options, callback) {
	return fs.readFile(path, 'utf8', function (err, str) {
		if (err) {
			return callback(err);
		}
		// console.log(options);
		return callback(null, markdown.toHTML(str));
	});
});

app.use(helmet({ contentSecurityPolicy: false }));
app.use(favicon(__dirname + '/assets/favicon.ico'));
app.use(logger("dev"));
app.use(compress());
app.use(bodyParser.json());

app.use(bodyParser.urlencoded({
	extended: false
}));

app.use(cookieParser());

app.use(express.static(path.join(__dirname, "assets"), { maxAge: 60 * 60 * 1000 }));

app.get("/", function (req, res, next) {
	fs.readFile('views/index.md', 'utf8', (err, str) => {
		if (err) {
			console.log(err);
			return next();
		}

		return res.render('index', {
			body: markdown.render(str)
		});
	});
});

app.get('/index.md', function (req, res, next) {
	fs.readFile('views/index.md', 'utf8', (err, str) => {
		if (err) {
			console.log(err);
			return next();
		}

		return res.send(markdown.render(str));
	});
});

app.get('/predict', function (req, res) {
	return res.render('predict', {
		peptide: ""
	});
});

app.post('/api/denovo', function (req, res) {
	let mgf = req.body.mgf;

	tmp.file({ discardDescriptor: true }, function _tempFileCreated(err, path, fd, cleanupCallback) {
		if (err) {
			console.log(err);
			res.status(500);
			return res.send(err);
		}

		console.log('path: ', path);

		fs.writeFile(path, mgf, err => {
			if (err) {
				console.log(err);
				res.status(500);
				res.send(err);

				return cleanupCallback();
			}

			const child = execFile('python', ['code/de.py', path], {maxBuffer: 1024 * 5000}, (err, stdout, stderr) => {
				if (err) {
					console.log(err);
					res.status(500);
					res.send(stderr);

					return cleanupCallback();
				}

				res.send(stdout);
				cleanupCallback();
			});
		})
	});
});

app.post('/predict', function (req, res) {
	return res.redirect("/predict/" + req.body.type + "/" + req.body.charge + "/" + req.body.peptide);
});

app.get(["/:url.html", "/:url"], function (req, res, next) {
	fs.access("views/" + req.params.url + ".ejs", fs.constants.R_OK, function (err) {
		if (err === null) {
			return res.render(req.params.url, {
				url: req.params.url,
				errMsg: ""
			});
		}
		else {
			return next();
		}
	});
});

app.get(["/:url.html", "/:url"], function (req, res, next) {
	fs.access("views/" + req.params.url + ".ejs", fs.constants.R_OK, function (err) {
		if (err === null) {
			return res.render(req.params.url, {
				url: req.params.url,
				errMsg: ""
			});
		}
		else {
			return next();
		}
	});
});

app.use(function (req, res, next) {
	res.status(404);
	res.render("404");
});

app.use(function (err, req, res, next) {
	res.status(err.status || 500);
	res.render("error.coffee", {
		message: err.message,
		error: err
	});
});

app.listen(app.get('port'), function () {
	var base;
	return console.log("Node running at localhost:" + (app.get('port')) + ", ENV is " + ((base = process.env).ENV != null ? base.ENV : base.ENV = 'productive'));
});

module.exports = app;
