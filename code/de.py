#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numba import int32, float32, float64, boolean
import numba as nb
import requests
import numpy as np
import pyteomics
from pyteomics import mgf, mass
from dataclasses import dataclass, asdict
import json

import sys

# mgf = ""
# for line in sys.stdin:
#     mgf += line

# print(mgf)


def openmgf(fn):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype="float32", use_index=False)
    return data


types = {"un": 0, "cid": 1, "etd": 2, "hcd": 3, "ethcd": 4, "etcid": 5}

cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}


def tojson(sps, charge=0, maxc=8, ignore=0):
    db = []

    for sp in sps:
        param = sp["params"]

        if not "charge" in param and ignore:
            continue
        c = int(str(param["charge"][0])[0])

        if charge > 0 and c != charge:
            continue
        if c < 1 or c > maxc:
            continue

        pep = title = ""

        if "title" in param:
            pep = title = param["title"]
        elif "seq" in param:
            pep = param["seq"]

        if "pepmass" in param:
            mass = param["pepmass"][0]
        else:
            mass = float(param["parent"])

        rtime = 0 if not "RTINSECONDS" in param else float(
            param["RTINSECONDS"])

        if "hcd" in param:
            try:
                hcd = param["hcd"]
                if hcd[-1] == "%":
                    hcd = float(hcd)
                elif hcd[-2:] == "eV":
                    hcd = float(hcd[:-2])
                    hcd = hcd * 500 * cr[c] / mass
                else:
                    raise Exception("Invalid type!")
            except:
                hcd = 0
        else:
            hcd = 0

        mz = sp["m/z array"]
        it = sp["intensity array"]

        db.append({"pep": pep, "charge": c, "mass": mass, "mz": mz, "it": it,
                   "nce": hcd, "title": title})

    return db


def readmgf(fn, idrst=None, c=0, **kws):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype="float32", use_index=False)

    codes = tojson(data, c, **kws)
    return codes


# In[4]:


class config(dict):
    def __init__(self, *args, **kwargs):
        super(config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def f4(x): return "{0:.4f}".format(x)


def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype="float32")
def np32(x): return np.array(x, dtype="float32")


def clipn(*kw, sigma=4):
    return np.clip(np.random.randn(*kw), -sigma, sigma) / sigma


def fastmass(pep, ion_type, charge, mod=None, cam=True):
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)

    if cam:
        base += 57.021 * pep.count("C") / charge

    if not mod is None:
        base += 15.995 * np.sum(mod == 1) / charge
    return base


def m1(pep, c=1, **kws): return fastmass(pep, ion_type="M", charge=c, **kws)


def ppmdiff(sp, pep):
    mass = fastmass(pep, "M", sp["charge"])
    return ((sp["mass"] - mass) / mass) * 1000000


def ppm(m1, m2):
    return ((m1 - m2) / m1) * 1000000


mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538, "Z": 147.0354,  # oxidaed M
        }
mono = {k: v for k, v in sorted(mono.items(), key=lambda item: item[1])}

ave_mass = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 160.1598, "E": 129.1155,
            "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
            "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782, "T": 101.1051,
            "W": 186.2132, "Y": 163.1760, "V": 99.1326}

amino_list = list("ACDEFGHIKLMNPQRSTVWYZ")
oh_dim = len(amino_list) + 3

amino2id = {"*": 0, "]": len(amino_list) + 1, "[": len(amino_list) + 2}
for i, a in enumerate(amino_list):
    amino2id[a] = i + 1

id2amino = {0: "*", len(amino_list) + 1: "]", len(amino_list) + 2: "["}
for a in amino_list:
    id2amino[amino2id[a]] = a

mass_list = asnp32([0] + [mono[a] for a in amino_list] + [0, 0])


def normalize(it, mode):
    if mode == 0:
        return it
    elif mode == 2:
        return np.sqrt(it)

    elif mode == 3:
        return np.sqrt(np.sqrt(it))

    elif mode == 4:
        return np.sqrt(np.sqrt(np.sqrt(it)))

    return it


def _remove_precursor(v, pre_mz, c, precision, low, r):
    for delta in (0, 1, 2):
        mz = pre_mz + delta / c
        if mz > 0 and mz >= low:
            pc = round((mz - low) / precision)

            if pc - r < len(v):
                v[max(0, pc - r): min(len(v), pc + r)] = 0
    return None  # force inline


def remove_precursor(v, pre_mz, c, precision, low, r=1):
    return _remove_precursor(v, pre_mz, c, precision, low, r)


def filterPeaks(v, _max_peaks):
    if _max_peaks <= 0 or len(v) <= _max_peaks:
        return v

    kth = len(v) - _max_peaks
    peak_thres = np.partition(v, kth)[kth]
    v[v < peak_thres] = 0
    return v


def flat(v, mz, it, pre, low, use_max):
    for i, x in enumerate(mz):
        pos = int(round((x - low) / pre))

        if pos < 0 or pos >= len(v):
            continue

        if use_max:
            v[pos] = max(v[pos], it[i])
        else:
            v[pos] += it[i]

    return v


def _vectorlize(mz, it, mass, c, precision, dim, low, mode, v, kth, th, de, dn, use_max):
    it /= np.max(it)

    if dn > 0:
        it[it < dn] = 0

    it = normalize(it, mode)  # pre-scale

    if kth > 0:
        it = filterPeaks(it, _max_peaks=kth)

    flat(v, mz, it, precision, low, use_max)

    if de == 1:
        _remove_precursor(v, mass, c, precision, low,
                          r=1)  # inplace, before scale

    v /= np.max(v)  # final scale, de can change max

    return v


def vectorlize(mz, it, mass, c, precision, dim, low, mode, out=None, kth=-1, th=-1, de=1, dn=-1, use_max=0):
    if out is None:
        out = np.zeros(dim, dtype="float32")
    return _vectorlize(asnp32(mz), np32(it), mass, c, precision, dim, low, mode, out, kth, th, de, dn, use_max)


def decode(seq2d):
    return np.int32([np.argmax(seq2d[i]) for i in range(len(seq2d))])


def topep(seq):
    return "".join(map(lambda n: id2amino[n], seq)).strip("*[]")


def toseq(pep):
    return np.int32([amino2id[c] for c in pep.upper()])


def what(seq2d):
    return topep(decode(seq2d))


def clean(pep):
    return pep.strip("*[]").replace("I", "L").replace("*", "L").replace("[", "A").replace("]", "R")


def iterate(x, bsz):
    while len(x) > bsz:
        yield x[:bsz]
        x = x[bsz:]
    yield x


# In[5]:


class hyper_para():
    @dataclass(frozen=True)
    class hyper():
        lmax: int = 30
        outlen: int = lmax + 2
        m1max: int = 2048
        mz_max: int = 2048
        pre: float = 0.1
        low: float = 0  # pre_denova / 2
        vdim: int = int(mz_max / pre)
        pdim: int = 512
        dim: int = vdim + 0
        mod: int = 0
        maxc: int = 8

        nnum: int = 4
        pnum: int = 4
        sp_dim: int = 4

        mode: int = 3
        scale: float = 0.3
        kth: int = 50
        cut_peaks: bool = True

        mix_mode = (2, 3)
        dynamic = config({"enhance": 1})

        inputs = config({
            "y": ([sp_dim, dim], "float32"),
            #             "sp_masks": ([hyper.dim], "int32"),
            "info": ([2], "float32"),
            "charge": ([8], "float32"),
            # "pks": ([2, pdim], "float32")
        })

        mtl = config({"peps": 1, "reg": 1, "pm": 0, "mass": 1, "peaks": 1, "charge": 1,
                      "pid": 0, "rk": 1, "exist": 1, "nums": 1, "di": 1, "nce": 0, "length": 1,
                      "ftype": 1, "rdrop": 0, "pep2": 0,
                      })

    def __init__(self):
        self.inner = self.__class__.hyper()

    def __getattr__(self, att):
        return getattr(self.inner, att)

    def dict(self):
        return asdict(self.inner)


class data_processor():
    def __init__(self, hyper):
        self.hyper = hyper

    def get_inputs(self, sps, training=0):
        hyper = self.hyper  # !
        batch_size = len(sps)

        inputs = config({})
        for spec in hyper.inputs:
            inputs[spec] = np.zeros(
                (batch_size, *hyper.inputs[spec][0]), dtype=hyper.inputs[spec][1])

        for i, sp in enumerate(sps):
            pep, mass, c, mzs, its = sp["pep"], sp["mass"], sp["charge"], sp["mz"], sp["it"]
            mzs = mzs / 1.00052

            its = normalize(its, self.hyper.mode)

            inputs.info[i][0] = mass / hyper.m1max
            inputs.info[i][1] = sp["type"]
            inputs.charge[i][c - 1] = 1

            mdim = min(hyper.dim - 1, round((mass * c - c + 1) / hyper.pre))
#             sp_masks[i][:mdim] = 1

            vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim,
                       hyper.low, 0, out=inputs.y[i][0], use_max=1)
    #         y[i][0] -= np.mean(y[i][0])
            inputs.y[i][1][:mdim] = inputs.y[i][0][:mdim][::-1]  # reverse it

            vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim,
                       hyper.low, 0, out=inputs.y[i][2], use_max=0)
    #         y[i][2] -= np.mean(y[i][2])
            inputs.y[i][3][:mdim] = inputs.y[i][2][:mdim][::-1]  # reverse mz

        return tuple([inputs[key] for key in inputs])


hyper = hyper_para()
processor = data_processor(hyper)


# In[6]:


def fix1(rst, mass, c, ppm=10):
    pscore = np.max(rst, axis=-1)
    seq = decode(rst)
    pep = topep(seq)
    seq = seq[:len(pep)]
    tol = mass * ppm / 1000000

    for i, char in enumerate(pep):
        if char in '*[]':
            pep = pep[:i]
            pscore[i:] = 1
            seq = seq[:i]
            break

    if len(pep) < 1:
        return "AAAAAK", -1, pscore

    msp = m1(topep(seq), c)
    delta = msp - mass
    pos = 0
    a = seq[0]

    if abs(delta) < tol:
        return topep(seq), -1, pscore

    for i in range(len(seq) - 1):  # no last pos
        mi = mass_list[seq[i]]
        for j in range(1, 21):
            if j == 8:
                continue  # ignore "I"

            d = msp - mass + (mass_list[j] - mi) / c

            if abs(d) < abs(delta):
                delta = d
                pos = i
                a = j

    if abs(delta) < tol:  # have good match
        candi = np.int32(seq == seq[pos])
        if np.sum(candi) > 1.5:  # ambiguis
            pos = np.argmin((1 - candi) * 10 + candi *
                            np.max(rst[:len(seq)], axis=-1))

        seq[pos] = a
        pscore[pos] = np.max(pscore)

        return topep(seq), pos, pscore
    else:
        return topep(seq), -1, pscore


# In[16]:


hyper = hyper_para()
processor = data_processor(hyper)


sp = readmgf(sys.argv[1])[0]
sp["type"] = types["hcd"]

rst = processor.get_inputs([sp])


# In[17]:


payload = {
    "instances": [
        {
            "sp_inp": rst[0][0].tolist(),  # np.zeros((4, 8192)).tolist(),
            "mz_inp": rst[1][0].tolist(),  # np.zeros((2, )).tolist(),
            "charge_inp": rst[2][0].tolist(),  # np.zeros((4, )).tolist(),
            # "pks_inp": rst[3][0].tolist(), #np.zeros((2, 512)).tolist(),
        }
    ]
}

# sending post request to TensorFlow Serving server
r = requests.post("http://localhost:6501/v1/models/novo_model:predict",
                  headers={"content-type": "application/json"},
                  json=payload)

try:
    pred = json.loads(r.content.decode("utf-8"))
    pep, _, matrix = fix1(pred["predictions"][0], sp["mass"], sp["charge"])
    pscore = matrix[:len(pep)]

    print({
        "pep": pep,
        "score": np.prod(pscore),
        "pscore": pscore.tolist(),
        "err": 0,
        "msg": ""
    })
except:
    print({
        "pep": "",
        "score": 0,
        "pscore": 0,
        "err": 1,
        "msg": r.content.decode('utf-8')
    })
