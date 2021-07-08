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


spikes = np.load("data/30_spikes_memory.npy")


#%%
TIME = np.arange(0, 1500, 0.001)
fr = get_network_firing_rates(spikes.T, 1000, len(spikes))

#%%

assembly = np.r_[6:12, 71:77]
background = np.r_[0:6, 12:71, 77:80]

fr_assembly = fr[assembly, :]
fr_background = fr[background, :]
mean_fr_background = np.mean(fr_background, axis=0)
std_fr_background = np.std(fr_background, axis=0)
mean_fr_assembly = np.mean(fr_assembly, axis=0)
std_fr_assembly = np.std(fr_assembly, axis=0)


interval1 = np.r_[590000:600000]
interval2 = np.r_[600000:610000]
interval3 = np.r_[1190000:1200000]
interval4 = np.r_[1200000:1201000]


mean_fr = np.zeros(shape=(4, 2))
mean_fr[0, 0] = np.mean(np.mean(fr_background[:, interval1], axis=1))
mean_fr[0, 1] = np.mean(np.mean(fr_assembly[:, interval1], axis=1))
mean_fr[1, 0] = np.mean(np.mean(fr_background[:, interval2], axis=1))
mean_fr[1, 1] = np.mean(np.mean(fr_assembly[:, interval2], axis=1))
mean_fr[2, 0] = np.mean(np.mean(fr_background[:, interval3], axis=1))
mean_fr[2, 1] = np.mean(np.mean(fr_assembly[:, interval3], axis=1))
mean_fr[3, 0] = np.mean(np.mean(fr_background[:, interval4], axis=1))
mean_fr[3, 1] = np.mean(np.mean(fr_assembly[:, interval4], axis=1))


#%%

fs = 13
ysc = 0.35

x1_1 = np.mean(fr_background[:, interval1], axis=1)
x2_1 = np.mean(fr_assembly[:, interval1], axis=1)
x1_2 = np.mean(fr_background[:, interval2], axis=1)
x2_2 = np.mean(fr_assembly[:, interval2], axis=1)
x1_3 = np.mean(fr_background[:, interval3], axis=1)
x2_3 = np.mean(fr_assembly[:, interval3], axis=1)
x1_4 = np.mean(fr_background[:, interval4], axis=1)
x2_4 = np.mean(fr_assembly[:, interval4], axis=1)


x_all = np.hstack((x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, x1_4, x2_4))


bins = np.histogram(x_all, bins=20)[1]


fig = plt.figure(figsize=(10, 2))
gs = fig.add_gridspec(1, 4, wspace=0.5, hspace=0.5)


ax = fig.add_subplot(gs[0, 0])
x1 = np.mean(fr_background[:, interval1], axis=1)
x2 = np.mean(fr_assembly[:, interval1], axis=1)
x_multi = [x1, x2]
c, d, e = ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    bins=bins,
    linewidth=0.5,
)

plt.setp(e[0], edgecolor="silver")
plt.setp(e[1], edgecolor="tab:orange")

ax.scatter(
    x=mean_fr[0, 0], y=ysc, marker="X", color="silver", ec="silver", s=70, linewidth=0.5
)
ax.scatter(
    x=mean_fr[0, 1],
    y=ysc,
    marker="X",
    color="tab:orange",
    ec="tab:orange",
    s=0,
    linewidth=0.5,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.set_ylabel("Density", fontsize=12)
ax.set_xlabel("Firing Rate (Hz)", fontsize=fs)
ax.tick_params(axis="both", labelsize=fs)


ax = fig.add_subplot(gs[0, 1])
x1 = np.mean(fr_background[:, interval2], axis=1)
x2 = np.mean(fr_assembly[:, interval2], axis=1)
x_multi = [x1, x2]
c, d, e = ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    ec="k",
    bins=bins,
    linewidth=0.5,
)
plt.setp(e[0], edgecolor="silver")
plt.setp(e[1], edgecolor="tab:orange")


ax.scatter(
    x=mean_fr[1, 0], y=ysc, marker="X", color="silver", ec="silver", s=75, linewidth=0.5
)
ax.scatter(
    x=mean_fr[1, 1],
    y=ysc,
    marker="X",
    color="tab:orange",
    ec="tab:orange",
    s=75,
    linewidth=0.5,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=fs)


ax = fig.add_subplot(gs[0, 2])
# x1 = np.mean(fr_background[:, 850000:860000], axis=1)
# x2 = np.mean(fr_assembly[:, 850000:860000], axis=1)
x1 = np.mean(fr_background[:, interval3], axis=1)
x2 = np.mean(fr_assembly[:, interval3], axis=1)
x_multi = [x1, x2]
c, d, e, = ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    ec="k",
    bins=bins,
    linewidth=0.5,
)

plt.setp(e[0], edgecolor="silver")
plt.setp(e[1], edgecolor="tab:orange")

ax.scatter(
    x=mean_fr[2, 0],
    y=ysc,
    marker="X",
    color="silver",
    ec="silver",
    s=75,
    linewidth=0.5,
)
ax.scatter(
    x=mean_fr[2, 1],
    y=ysc,
    marker="X",
    color="tab:orange",
    ec="tab:orange",
    s=75,
    linewidth=0.5,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=fs)


ax = fig.add_subplot(gs[0, 3])
x1 = np.mean(fr_background[:, interval4], axis=1)
x2 = np.mean(fr_assembly[:, interval4], axis=1)
x_multi = [x1, x2]
c, d, e = ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    ec="k",
    bins=bins,
    linewidth=0.5,
)

plt.setp(e[0], edgecolor="silver")
plt.setp(e[1], edgecolor="tab:orange")


ax.scatter(
    x=mean_fr[3, 0], y=ysc, marker="X", color="silver", ec="silver", s=75, linewidth=0.5
)
ax.scatter(
    x=mean_fr[3, 1],
    y=ysc,
    marker="X",
    color="tab:orange",
    ec="tab:orange",
    s=75,
    linewidth=0.5,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=fs)


plt.savefig("memory_hist.png", bbox_inches="tight", dpi=500)


from brokenaxes import brokenaxes

range1 = np.r_[580000:820000]
range2 = np.r_[1180000:1220000]

fig = plt.figure(figsize=(10, 2))
bax = brokenaxes(xlims=((580, 820), (1180, 1220)), wspace=0.05)
x = np.linspace(0, 1, 100)
bax.plot([590, 600], [55, 55], color="k")
bax.plot([1190, 1200], [55, 55], color="k")


bax.plot(
    TIME[range1][::1000],
    mean_fr_background[range1][::1000],
    label="background",
    color="grey",
)
bax.fill_between(
    TIME[range1][::1000],
    mean_fr_background[range1][::1000] - std_fr_background[range1][::1000],
    mean_fr_background[range1][::1000] + std_fr_background[range1][::1000],
    alpha=0.2,
    color="grey",
)
bax.plot(
    TIME[range1][::1000],
    mean_fr_assembly[range1][::1000],
    label="assembly",
    color="tab:orange",
)
bax.fill_between(
    TIME[range1][::1000],
    mean_fr_assembly[range1][::1000] - std_fr_assembly[range1][::1000],
    mean_fr_assembly[range1][::1000] + std_fr_assembly[range1][::1000],
    alpha=0.2,
    color="tab:orange",
)
bax.plot(TIME[range2][::1000], mean_fr_background[range2][::1000], color="grey")
bax.fill_between(
    TIME[range2][::1000],
    mean_fr_background[range2][::1000] - std_fr_background[range2][::1000],
    mean_fr_background[range2][::1000] + std_fr_background[range2][::1000],
    alpha=0.2,
    color="grey",
)
bax.plot(TIME[range2][::1000], mean_fr_assembly[range2][::1000], color="tab:orange")
bax.fill_between(
    TIME[range2][::1000],
    mean_fr_assembly[range2][::1000] - std_fr_assembly[range2][::1000],
    mean_fr_assembly[range2][::1000] + std_fr_assembly[range2][::1000],
    alpha=0.2,
    color="tab:orange",
)


bax.set_xlabel("Time (s)", fontsize=fs)
bax.set_ylabel("Firing rate (Hz)", fontsize=fs)
bax.tick_params(axis="both", labelsize=fs)


plt.savefig("memory.png", bbox_inches="tight", dpi=500)

# %%
