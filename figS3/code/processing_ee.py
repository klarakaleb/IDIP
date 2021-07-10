#%%
import numpy as np
from numba import jit

spars = np.arange(0.0, 1.1, 0.1)

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


spars = np.round(spars, 2)


@jit(nopython=True)
def get_network_firing_rates(spikes, timewindow, len_time):
    firing_rate = np.zeros(shape=(len(spikes), len_time))
    for i in range(0, len_time):
        if i + timewindow < len_time:
            firing_rate[:, i] = np.sum(spikes[:, i : i + timewindow], axis=1) / (
                timewindow / 1000
            )
    return firing_rate


fr_mat = np.zeros(shape=(len(spars), 80, len(seeds)))
fr_mat2 = np.zeros(shape=(len(spars), 80, len(seeds)))

path = '../data/original/ee/'


for si, s in enumerate(spars):
    print(si)
    for ssi, ss in enumerate(seeds):
        print(ss)
        # spikes = np.load("data2000/" + str(s) + "_" + str(ss) + "_spikes.npy")
        # fr = get_network_firing_rates(spikes.T, 1000, len(spikes))
        fr_mat[si, :, ssi] = np.load(
            path + str(s) + "_" + str(ss) + "_fr_end.npy"
        )
        fr_mat2[si, :, ssi] = np.load(
            path + str(s) + "_" + str(ss) + "_fr_init.npy"
        )

print(fr_mat)
np.save(path + "fr_mat.npy", fr_mat)
np.save(path + "fr_mat2.npy", fr_mat2)
