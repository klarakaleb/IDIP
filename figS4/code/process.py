import numpy as np
from numba import jit
import scipy.stats

dirs = [
    123,
    167,
    23,
    30,
    43,
    56,
    57,
    77,
    78,
    230,
    234,
    27,
    30,
    39,
    45,
    456,
    536,
    55,
    73,
    893,
]
cc = np.zeros(shape=(len(dirs), 133))

path = '../data/'

count = 0

for i in range(len(dirs)):
    print(i)
    cc_i = np.load(path + str(dirs[i]) + "_cc.npy")
    #np.save(str(dirs[i]) + "_cc1000.npy", cc_i)
    cc[count] = cc_i
    count += 1

print(np.mean(cc, axis=0))

np.save(path + "cc.npy", cc)