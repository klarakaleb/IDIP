#%%
import numpy as np
import matplotlib.pyplot as plt

fr_silent = np.load("data/remap/fr_silent.npy")
fr_active = np.load("data/remap/fr_active.npy")


fs = 13

x = np.arange(0, 90, 0.001)
y1 = np.mean(
    fr_active[
        :,
    ],
    axis=0,
)
y2 = np.mean(
    fr_silent[
        :,
    ],
    axis=0,
)

y1_err = np.std(fr_active[:], axis=0)
y2_err = np.std(fr_silent[:], axis=0)

fig = plt.figure(figsize=(10, 7.0))
gs = fig.add_gridspec(3, 4, wspace=0.8, hspace=0.75)


ax = fig.add_subplot(gs[0, 3])
# ax = fig.add_subplot(gs[0,3])
# ax = plt.axes()
# ax.plot(x[30000-1001:33000-2000],y1[30000-1001:33000-2000],color='tab:green')
ax.plot(x[30000 - 1001 : 33000 - 2000], y2[30000 - 1001 : 33000 - 2000], color="grey")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# plt.fill_between(x[30000-2000:33000-2000],y1[30000-2000:33000-2000]-y1_err[30000-2000:35000-2000],y1[30000-2000:35000-2000]+y1_err[30000-2000:35000-2000],color='tab:green',alpha = 0.2)
plt.fill_between(
    x[30000 - 1001 : 33000 - 2000],
    y2[30000 - 1001 : 33000 - 2000] - y2_err[30000 - 1001 : 33000 - 2000],
    y2[30000 - 1001 : 33000 - 2000] + y2_err[30000 - 1001 : 33000 - 2000],
    color="grey",
    alpha=0.2,
)
ax.set_ylim(0, 4.5)
# x.set_xticks(np.arange(10, 80+1, 80))
ax.tick_params(axis="both", which="major", labelsize=fs - 1)
ax.set_ylabel("Firing Rate (Hz)", fontsize=fs, labelpad=10)
ax.set_xlabel("Time (s)", fontsize=fs)
ax.fill_betweenx(range(5), 30 - 1, 33 - 2, alpha=0.2, color="yellow")


ax = fig.add_subplot(gs[1, :3])
# ax = fig.add_subplot(gs[0,3])
# ax = plt.axes()
ax.plot(x[10000:80000], y1[10000:80000], color="tab:green")
ax.plot(x[10000:80000], y2[10000:80000], color="grey")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.fill_between(
    x[10000:80000],
    y1[10000:80000] - y1_err[10000:80000],
    y1[10000:80000] + y1_err[10000:80000],
    color="tab:green",
    alpha=0.2,
)
plt.fill_between(
    x[10000:80000],
    y2[10000:80000] - y2_err[10000:80000],
    y2[10000:80000] + y2_err[10000:80000],
    color="grey",
    alpha=0.2,
)
ax.fill_betweenx(range(5), 30 - 1, 60 - 2, alpha=0.2, color="yellow")
ax.set_ylim(0, 4.5)
ax.set_ylabel("Firing Rate (Hz)", fontsize=fs, labelpad=10)
ax.set_xlabel("Time (s)", fontsize=fs)
# x.set_xticks(np.arange(10, 80+1, 80))
ax.tick_params(axis="both", which="major", labelsize=fs - 1)
# ratio = 1#0.2#0.4
# xleft, xright = ax.get_xlim()
# ybottom, ytop = ax.get_ylim()
# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

y1 = np.mean(fr_active[:][:, 0 : (30000 - 1000)])
y2 = np.mean(fr_silent[:][:, (30000 - 1000) : (60000 - 2000)])

ax = fig.add_subplot(gs[1, 3])
for i in range(len(fr_silent)):
    ax.plot(
        2,
        np.mean(fr_silent[:][i, (30000 - 1000) : (60000 - 2000)]),
        marker=".",
        color="darkgrey",
    )

for i in range(len(fr_active)):
    ax.plot(
        1, np.mean(fr_active[:][i, 0 : (30000 - 1000)]), marker=".", color="darkgrey"
    )

for i in range(len(fr_active)):
    ax.plot(
        [1, 2],
        [
            np.mean(fr_active[:][i, 0 : (30000 - 1000)]),
            np.mean(fr_silent[:][i, (30000 - 1000) : (60000 - 2000)]),
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.tick_params(axis="both", which="major", labelsize=fs - 1)
ax.plot(1, y1, marker="o", color="black")
ax.plot(2, y2, marker="o", color="black")
ax.plot([1, 2], [y1, y2], "-", color="black")
ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_ylabel("Firing Rate (Hz)", fontsize=fs, labelpad=10)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)


fr_silent = np.load("data/remap/fr_silent_ni.npy")
fr_active = np.load("data/remap/fr_active_ni.npy")


x = np.arange(0, 90, 0.001)
y1 = np.mean(
    fr_active[
        :,
    ],
    axis=0,
)
y2 = np.mean(
    fr_silent[
        :,
    ],
    axis=0,
)

y1_err = np.std(fr_active[:], axis=0)
y2_err = np.std(fr_silent[:], axis=0)

ax = fig.add_subplot(gs[2, :3])
# ax = fig.add_subplot(gs[0,3])
# ax = plt.axes()
ax.plot(x[10000:80000], y1[10000:80000], color="tab:green")
ax.plot(x[10000:80000], y2[10000:80000], color="grey")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.fill_between(
    x[10000:80000],
    y1[10000:80000] - y1_err[10000:80000],
    y1[10000:80000] + y1_err[10000:80000],
    color="tab:green",
    alpha=0.2,
)
plt.fill_between(
    x[10000:80000],
    y2[10000:80000] - y2_err[10000:80000],
    y2[10000:80000] + y2_err[10000:80000],
    color="grey",
    alpha=0.2,
)
ax.fill_betweenx(range(5), 30 - 1, 60 - 2, alpha=0.2, color="yellow")
ax.set_ylim(0, 4.5)
ax.set_ylabel("Firing Rate (Hz)", fontsize=fs, labelpad=10)
ax.set_xlabel("Time (s)", fontsize=fs)
ax.tick_params(axis="both", which="major", labelsize=fs - 1)
# ratio = 1#0.2#0.4
# xleft, xright = ax.get_xlim()
# ybottom, ytop = ax.get_ylim()
# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

y1 = np.mean(fr_active[:][:, 0 : (30000 - 1000)])
y2 = np.mean(fr_silent[:][:, (30000 - 1000) : (60000 - 2000)])

ax = fig.add_subplot(gs[2, 3])
for i in range(len(fr_silent)):
    ax.plot(
        2,
        np.mean(fr_silent[:][i, (30000 - 1000) : (60000 - 2000)]),
        marker=".",
        color="darkgrey",
    )

for i in range(len(fr_active)):
    ax.plot(
        1, np.mean(fr_active[:][i, 0 : (30000 - 1000)]), marker=".", color="darkgrey"
    )

for i in range(len(fr_active)):
    ax.plot(
        [1, 2],
        [
            np.mean(fr_active[:][i, 0 : (30000 - 1000)]),
            np.mean(fr_silent[:][i, (30000 - 1000) : (60000 - 2000)]),
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, y1, marker="o", color="black")
ax.plot(2, y2, marker="o", color="black")
ax.plot([1, 2], [y1, y2], "-", color="black")
ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_ylabel("Firing Rate (Hz)", fontsize=fs, labelpad=10)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=fs - 1)


plt.savefig("silencing_grid.png", dpi=500, bbox_inches="tight")

# %%

