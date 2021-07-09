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


cc = np.load("data/cc.npy")
cc2 = np.load("data/vogels/cc.npy")


mean_cc = np.mean(cc, axis=0)
mean_cc2 = np.mean(cc2, axis=0)

std_cc = np.std(cc, axis=0)
std_cc2 = np.std(cc2, axis=0)


#%%

# processing
TIME = np.arange(0, 600, 0.001)
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


hetero = np.load("data/hetero/IDIP_hetero_repeats1.npy")[:, : len(TIME)]
hetero2 = np.load("data/hetero/IDIP_hetero_repeats2.npy")[:, : len(TIME)]

mean_hetero = np.mean(hetero, axis=0)
std_hetero = np.std(hetero, axis=0)
mean_hetero2 = np.mean(hetero2, axis=0)
std_hetero2 = np.std(hetero2, axis=0)


mean_fr_hetero = np.load("data/hetero/mean_fr.npy")
mean_fr = np.load("data/hetero/mean_fr_original.npy")


#%%
fs = 13

fig = plt.figure(figsize=(5.5, 6.5))
gs = fig.add_gridspec(2, 2, wspace=0.5, hspace=0.6)

ax = fig.add_subplot(gs[0, :1])
ax.plot(
    TIME[0:-1001][::1000],
    mean_hetero[0:-1001][::1000],
    color="grey",
    label="Firing rate",
)
ax.fill_between(
    TIME[0:-1001][::1000],
    mean_hetero[0:-1001][::1000] - std_hetero[0:-1001][::1000],
    mean_hetero[0:-1001][::1000] + std_hetero[0:-1001][::1000],
    alpha=0.2,
    color="grey",
)
ax.plot(
    TIME[0:-1001][::1000],
    mean_hetero2[0:-1001][::1000],
    color="#ffb347",
    label="Firing rate",
)
ax.fill_between(
    TIME[0:-1001][::1000],
    mean_hetero2[0:-1001][::1000] - std_hetero2[0:-1001][::1000],
    mean_hetero2[0:-1001][::1000] + std_hetero2[0:-1001][::1000],
    alpha=0.2,
    color="#ffb347",
)
ax.set_ylabel("Firing Rate (Hz)", fontsize=fs)
ax.set_xlabel("Time (s)", fontsize=fs)
ax.tick_params(axis="both", labelsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = fig.add_subplot(gs[0, 1])
for i in range(len(mean_fr)):
    ax.plot(2, mean_fr_hetero[i], marker=".", color="darkgrey")

for i in range(len(mean_fr)):
    ax.plot(1, mean_fr[i], marker=".", color="darkgrey")

for i in range(len(mean_fr)):
    ax.plot([1, 2], [mean_fr[i], mean_fr_hetero[i]], "-", color="darkgrey", lw=0.5)

y1 = np.mean(mean_fr)
y2 = np.mean(mean_fr_hetero)

ax.plot(1, y1, marker="o", color="black")
ax.plot(2, y2, marker="o", color="black")
ax.plot([1, 2], [y1, y2], "-", color="black")
ax.set_ylim(2, 8)
ax.set_xlim(0.5, 2.5)
ax.set_ylabel("Firing Rate (Hz)", fontsize=fs)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=fs)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ratio = 1#0.2#0.4
# xleft, xright = ax.get_xlim()
# ybottom, ytop = ax.get_ylim()
# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

ax.tick_params(axis="both", labelsize=fs)


ax = fig.add_subplot(gs[1, :2])
plt.errorbar(
    range(133)[::5], mean_cc[::5], yerr=std_cc[::5], fmt="ok", label="IDIP", ms=4.0
)
plt.errorbar(
    range(133)[::5],
    mean_cc2[::5],
    yerr=std_cc2[::5],
    fmt="o",
    color="grey",
    label="iSTDP",
    ms=4.0,
)


ax.plot(range(133), np.repeat(0, 133), "k--")
ax.set_ylabel("Correlation Coefficient", fontsize=fs)
ax.set_xlabel("Time bin", fontsize=fs)
ax.set_ylim(-0.1, 1.1)
ax.tick_params(axis="both", labelsize=fs)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
leg = ax.legend(ncol=1, bbox_to_anchor=(0.9, 1.0))


plt.savefig("hetero_grid.png", bbox_inches="tight", dpi=500)

# %%
