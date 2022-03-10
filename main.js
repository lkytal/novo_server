
process.env.ENV = process.env.ENV || 'production';

if (process.env.ENV == 'production') {
	process.env.NODE_ENV = 'production';
}

require('./app.js');
