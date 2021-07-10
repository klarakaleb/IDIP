#%%
import numpy as np
import scipy.stats
from numba import jit


lrs = [0.01, 0.1, 1.0, 10.0]
lrs = np.round(lrs, 2)

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


@jit(nopython=True)
def get_network_firing_rates(spikes, timewindow, len_time):
    firing_rate = np.zeros(shape=(len(spikes), len_time))
    for i in range(0, len_time):
        if i + timewindow < len_time:
            firing_rate[:, i] = np.sum(spikes[:, i : i + timewindow], axis=1) / (
                timewindow / 1000
            )
    return firing_rate


fr_mat = np.zeros(shape=(len(lrs), 2000, len(seeds)))
std_mat = np.zeros(shape=(len(lrs), 2000, len(seeds)))
cc = np.zeros(shape=(len(lrs), 133, len(seeds)))

isicv = np.zeros(shape=(len(lrs), 80, len(seeds)))
fr_end = np.zeros(shape=(len(lrs), 80, len(seeds)))

path = '../data/original/lr/'



for lri, lr in enumerate(lrs):
    print(lr)
    for si, s in enumerate(seeds):
        print(s)
        spikes = np.load(path + str(lr) + "_" + str(s) + "_spikes.npy")
        fr = get_network_firing_rates(spikes.T, 1000, len(spikes))
        fr_mat[lri, :, si] = fr[:80].mean(axis=0)[::1000]
        std_mat[lri, :, si] = fr[:80].std(axis=0)[::1000]
        for i in range(133):
            cc[lri, i, si] = scipy.stats.spearmanr(
                np.mean(fr[0:80, 0:15000], axis=1),
                np.mean(fr[0:80, (i * 15000) : ((i + 1) * 15000)], axis=1),
            )[0]
        fr_end[lri, :, si] = fr[:80, -16001:-1001].mean(axis=1)
        isi_cv = []
        for i in range(80):
            indices = np.where(spikes.T[i, -16001:-1001] == 1)  # last 15 seconds
            timings = TIME[indices]
            diffs = np.diff(timings)
            isi_cv.append(np.std(diffs) / np.mean(diffs))
        isicv[lri, :, si] = isi_cv

np.save(path + "fr_mat.npy", fr_mat)
np.save(path + "std_mat.npy", std_mat)
np.save(path + "cc_mat.npy", cc)
np.save(path + "isi_cv.npy", isicv)
np.save(path + "fr_end.npy", fr_end)
