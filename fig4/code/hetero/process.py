import numpy as np
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



TIME = np.arange(0, 2000, 0.001)
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


mean_fr = np.zeros(shape=(len(dirs), len(TIME)))
mean_fr_hetero = np.zeros(shape=(len(dirs), len(TIME)))


mean_fr_end = np.zeros(shape=(len(dirs)))
mean_fr_end_nh = np.zeros(shape=(len(dirs)))


count = 0
for d in dirs:
    print(d)
    spikes = np.load("../data/hetero/" + str(d) + "_spikes.npy")
    fr = get_network_firing_rates(spikes.T, 1000, len(spikes))
    mean_fr[count] = np.mean(fr[np.r_[0:7, 8:77, 78:80]], axis=0)
    mean_fr_hetero[count] = np.mean(fr[np.r_[7, 77]], axis=0)
    mean_fr_end[count] = np.mean(fr[0:80, -11001:-1001])
    spikes = np.load("../../fig3/data/" + str(d) + "_spikes.npy")
    fr = get_network_firing_rates(spikes.T, 1000, len(spikes))
    mean_fr_end_nh[count] = np.mean(fr[0:80, -11001:-1001])
    count += 1

np.save("hetero_repeats1.npy", mean_fr)
np.save("hetero_repeats2.npy", mean_fr_hetero)
np.save("mean_fr_end.npy", mean_fr_end)
np.save("mean_fr_end_nh.npy", mean_fr_end_nh)




