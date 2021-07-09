#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def get_network_firing_rates(spikes, timewindow, len_time):
    firing_rate = np.zeros(shape=(len(spikes), len_time))
    for i in range(0, len_time):
        if i + timewindow < len_time:
            firing_rate[:, i] = np.sum(spikes[:, i : i + timewindow], axis=1) / (
                timewindow / 1000
            )
    return firing_rate


TIME = np.arange(0, 600, 0.001)
dirs = [
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
mean_fr = np.zeros(shape=(len(dirs)))

path = '../data/hetero/'

count = 0
for d in dirs:
    print(d)
    spikes = np.load(path + str(d) + "_spikes.npy")
    fr = get_network_firing_rates(spikes.T, 1000, len(spikes))[:,:600000]
    mean_fr[count] = np.mean(fr[0:80, -11001:-1001])
    print(mean_fr[count])
    count += 1


np.save(path+"mean_fr.npy", mean_fr)

count = 0
for d in dirs:
    print(d)
    spikes = np.load("../../fig3/data/" + str(d) + "_spikes.npy")
    fr = get_network_firing_rates(spikes.T, 1000, len(spikes))[:,:600000]
    mean_fr[count] = np.mean(fr[0:80, -11001:-1001])
    print(mean_fr[count])
    count += 1


np.save(path+"mean_fr_original.npy", mean_fr)