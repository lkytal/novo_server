#noCompile

doctype 5
html lang: "zh-CN", ->
	head ->
		title @title
		meta charset: "utf-8"
		meta name: "viewport", content: "width=device-width, initial-scale=1.0"
		meta name: "author", content: "lkytal"
		link rel: "shortcut icon", href: "favicon.ico"
		link rel: "stylesheet", href: "/css/bootstrap.min.css"
		ie "lte IE 7", ->
			link rel: "stylesheet", type: "text/css", href: "/css/ie7.css"
		ie "lte IE 6", ->
			link rel: "stylesheet", type: "text/css", href: "/css/ie6.min.css"
		ie "lt IE 9", ->
			script src: "/js/html5shiv.js"
			script src: "/js/respond.min.js"

	body ->
		div ".container", ->
			h1 "#{@error.status} : #{@message}"
			br
			pre @error.stack
