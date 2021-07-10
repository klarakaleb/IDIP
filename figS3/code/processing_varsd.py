import numpy as np
from numba import jit

spars = np.arange(0.0, 0.55, 0.05)

seeds = [
    30,
    43,
    56,
    77,
    123,
    78,
    57,
    23,
    167,
    333,
    45,
    27,
    893,
    456,
    73,
    55,
    234,
    536,
    230,
    39,
]

spars = np.round(spars, 3)
TIME = np.arange(0, 2000, 0.001)


fr_mat = np.zeros(shape=(len(spars), 80, len(seeds)))
fr_mat2 = np.zeros(shape=(len(spars), 80, len(seeds)))
isicv = np.zeros(shape=(len(spars), 80, len(seeds)))

path = '../data/original/varsd/'


for si, s in enumerate(spars):
    for ssi, ss in enumerate(seeds):
        print(s)
        fr_mat[si, :, ssi] = np.load(path + str(s) + "_" + str(ss) + "_fr_end.npy")
        fr_mat2[si, :, ssi] = np.load(path + str(s) + "_" + str(ss) + "_fr_init.npy")
        spikes = np.load(path + str(s) + "_" + str(ss) + "_spikes.npy")
        isi_cv = []
        for i in range(80):
            indices = np.where(spikes.T[i, -16001:-1001] == 1)  # last 15 seconds
            timings = TIME[indices]
            diffs = np.diff(timings)
            isi_cv.append(np.std(diffs) / np.mean(diffs))
        isicv[si, :, ssi] = isi_cv

print(fr_mat)
np.save(path + "fr_mat.npy", fr_mat)
np.save(path + "fr_mat2.npy", fr_mat2)
np.save(path + "isicv.npy", isicv)