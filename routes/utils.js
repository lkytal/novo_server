
mono = {
	"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
	"C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
	"E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
	"Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538
}

ave_mass = {
	"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 160.1598, "E": 129.1155,
	"Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
	"M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782, "T": 101.1051,
	"W": 186.2132, "Y": 163.1760, "V": 99.1326
}

let charMap = {
	'@': 0, '[': 21, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
	'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
	'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

let max_len = 30;
let max_in = max_len + 2;
let x_dim = 23

let precision = 0.1
let low = 0
let dim = 20000
let upper = Math.floor(low + dim * precision)
let mz_scale = 3000.0
let max_mz = dim * precision + low

function fastmass(pep, ion_type = 'M', charge = 1) {
	pep = pep.toUpperCase();

	let mass = 0;

	for (let a of pep) {
		mass += mono[a];

		if (a == 'C') mass += 57.021;
	}

	mass += 18; //residual

	if (ion_type == 'M') {
		mass = (mass + 1.008 * charge);
	}
	else if (ion_type == 'b') {
		mass = mass + 1.008 * charge - 18.01;
	}
	else if (ion_type == 'y') {
		mass = mass + 1.008 * charge;
	}

	return mass / charge;
}

Array.prototype.at = function (i) {
	if (i < 0) {
		return this[this.length + i];
	}
	else
		return this[i];
}

function matrix(x, y) {
	let m = new Array();

	for (let i = 0; i < x; i++) {
		m[i] = new Array();

		for (let j = 0; j < y; j++) {
			m[i][j] = 0;
		}
	}

	return m;
}

function embed(type, charge, pep, nce='25') {
	pep = pep.toUpperCase();
	type = parseInt(type);
	charge = parseInt(charge);
	nce = parseInt(nce);

	let em = matrix(max_in, x_dim);

	if (pep.length > max_len) return false;

	em[pep.length][21] = 1; // ending pos, next line with +1 to skip this
	for (let i = pep.length + 1; i < max_in - 1; i++)
		em[i][0] = 1; // padding first, meta column should not be affected

	let meta = em[em.length - 1];
	meta[0] = fastmass(pep, ion_type = 'M', charge = 1) / max_mz; // pos 0, overwrtie padding
	meta[charge] = 1; // pos 1 - 4
	meta[5 + type] = 1; // pos 5 - 8
	meta[-1] = nce / 100.0

	for (let i = 0; i < pep.length; i++) {
		em[i][charMap[pep[i]]] = 1; // 1 - 20
		em[i][x_dim - 1] = mono[pep[i]] / max_mz;
	}

	return em;
}

function filter(y, low = 0.0, pre = 0.1, th = 0.1, power = 1) {
	const max = Math.max(...y);

	for (let i = 0; i < y.length; i++) {
		y[i] = (y[i] / max) ** power;
	}

	let mzs = [];
	let its = [];

	for (let i = 0; i < y.length; i++) {
		if (y[i] > th ** power) {
			mzs.push(low + pre * i);
			its.push(y[i]);
		}
	}

	return [mzs, its];
}

exports.filter = filter;
exports.embed = embed;
exports.fastmass = fastmass;

// let p = 'AAAAPEPSE';
// console.log(fastmass(p, 'M', 1));
// console.log(embed(3, 2, 'PEPTIDE'));
