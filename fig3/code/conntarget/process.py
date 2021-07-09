import numpy as np
from numba import jit

targets = np.arange(4.0, 8.1, 0.1)
connect = np.arange(0.5, 10.5, 0.5)

seeds = [30, 43, 56, 77, 45]


targets = np.round(targets, 1)
connect = np.round(connect, 1)


fr_mat = np.zeros(shape=(len(targets), len(connect), len(seeds)))
sd_mat = np.zeros(shape=(len(targets), len(connect), len(seeds)))


for si, s in enumerate(seeds):
    for ti, t in enumerate(targets):
        print(t)
        for ci, c in enumerate(connect):
            print(c)
            fr = np.load(
                "../data/"
                + str(s)
                + "_"
                + str(t)
                + "_"
                + str(c)
                + "_fr_end.npy"
            )
            fr_mat[ti, ci, si] = fr.mean()
            sd_mat[ti, ci, si] = fr.std()

print(fr_mat)

np.save("fr_mat.npy", fr_mat)
np.save("sd_mat.npy", sd_mat)
