let gulp = require('gulp');
let plumber = require('gulp-plumber');
let coffee = require('gulp-coffee');
let watch = require('gulp-watch');
let babel = require("gulp-babel");
let uglify = require('gulp-uglify');
let nodemon = require('gulp-nodemon');
let less = require('gulp-less');
let cleanCSS = require('gulp-clean-css');
let autoprefixer = require('gulp-autoprefixer');
let postcss = require('gulp-postcss');
let color_rgba_fallback = require('postcss-color-rgba-fallback');
let opacity = require('postcss-opacity');
let pseudoelements = require('postcss-pseudoelements');
let vmin = require('postcss-vmin');
let pixrem = require('pixrem');
let will_change = require('postcss-will-change');
let processors = [will_change, color_rgba_fallback, opacity, pseudoelements, vmin, pixrem];

let coffeeToJs = function (src, dest) {
    console.log("complie " + src);
    if (dest == null) {
        dest = './';
    }
    return gulp.src(src).pipe(plumber()).on('error', console.log).pipe(coffee({
        bare: true
    })).pipe(babel()).pipe(uglify()).pipe(gulp.dest(dest));
};

let watchCoffee = function (src, dest) {
    if (dest == null) {
        dest = './';
    }
    return gulp.src(src).pipe(watch(src)).pipe(plumber()).on('error', console.log).pipe(coffee({
        bare: true
    })).pipe(babel()).pipe(uglify()).pipe(gulp.dest(dest));
};

gulp.task('coffee', function () {
    return coffeeToJs('./Code/*.coffee', './assets/js');
});

let lessToJs = function (src, dest) {
    console.log("complie " + src);
    if (dest == null) {
        dest = './';
    }
    return gulp.src(src).pipe(plumber()).on('error', console.log).pipe(less()).pipe(autoprefixer()).pipe(postcss(processors)).pipe(cleanCSS()).pipe(gulp.dest(dest));
};

let watchLess = function (src, dest) {
    if (dest == null) {
        dest = './';
    }
    return gulp.src(src).pipe(watch(src)).pipe(plumber()).on('error', console.log).pipe(less()).pipe(cleanCSS()).pipe(autoprefixer()).pipe(postcss(processors)).pipe(gulp.dest(dest));
};

gulp.task('less', function () {
    return lessToJs('./less/**/*.*', './assets/css');
});

let watchJS = function (src, dest) {
    if (dest == null) {
        dest = './assets/js';
    }
    return gulp.src(src).pipe(watch(src)).pipe(plumber()).on('error', console.log).pipe(babel()).pipe(uglify()).pipe(gulp.dest(dest));
};

gulp.task('watch', function () {
    watchCoffee('./Code/*.coffee', './assets/js');
    watchLess('./less/**/*.*', './assets/css');
    return watchJS('./Code/*.js', './assets/js');
});

gulp.task('nodemon', gulp.series('coffee', 'less', function (cb) {
    process.env.ENV = 'localWeb';
    return nodemon({
        script: 'main.js',
        watch: ['app.coffee', 'app.js', "routes/"],
        ignore: ["code/"]
    }).once('start', cb);
}));

gulp.task('web', gulp.series('nodemon', function () {
    var browserSync;
    browserSync = require('browser-sync').create();
    return browserSync.init(null, {
        proxy: "http://localhost:6000",
        files: ["views/*.*", "assets/**/*.*"],
        browser: "ff",
        port: 7000
    });
}));

gulp.task('default', gulp.parallel('web', 'watch'));
