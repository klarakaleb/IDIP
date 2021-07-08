#%%
import numpy as np
import matplotlib.pyplot as plt

fr_silent = np.load("../hippocampus/fig2/remap/fr_silent.npy")
fr_active = np.load("../hippocampus/fig2/remap/fr_active.npy")

fr_silent_v1 = np.load("v1/remap/fr_silent.npy")
fr_active_v1 = np.load("v1/remap/fr_active.npy")


fr_silent_v2 = np.load("v2/remap/fr_silent.npy")
fr_active_v2 = np.load("v2/remap/fr_active.npy")


fr_silent_v3 = np.load("v3/remap/fr_silent.npy")
fr_active_v3 = np.load("v3/remap/fr_active.npy")


fr_silent_v4 = np.load("v4/remap/fr_silent.npy")
fr_active_v4 = np.load("v4/remap/fr_active.npy")


fr_silent_v5 = np.load("v5/remap/fr_silent.npy")
fr_active_v5 = np.load("v5/remap/fr_active.npy")


fs = 13

colors = [
    "black",
    "grey",
    "tab:green",
    "tab:orange",
    "tab:blue",
    "tab:purple",
    "tab:red",
]


fig = plt.figure(figsize=(10, 3))
gs = fig.add_gridspec(1, 5, wspace=0.5, hspace=0.5)

ax = fig.add_subplot(gs[0])
fr_silent = fr_silent_v1
fr_active = fr_active_v1

y1 = np.mean(fr_active[:][:, 0 : (300 - 10)])
y2 = np.mean(fr_silent[:][:, (300 - 10) : (600 - 20)])


for i in range(len(fr_silent)):
    ax.plot(
        2,
        np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        marker=".",
        color="darkgrey",
    )

for i in range(len(fr_active)):
    ax.plot(1, np.mean(fr_active[:][i, 0 : (300 - 10)]), marker=".", color="darkgrey")

for i in range(len(fr_active)):
    ax.plot(
        [1, 2],
        [
            np.mean(fr_active[:][i, 0 : (300 - 10)]),
            np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, y1, marker="o", color=colors[2])
ax.plot(2, y2, marker="o", color=colors[2])
ax.plot([1, 2], [y1, y2], "-", color=colors[2])

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
ax.set_title("v1", color=colors[2], fontsize=fs)

ax = fig.add_subplot(gs[1])
fr_silent = fr_silent_v2
fr_active = fr_active_v2

y1 = np.mean(fr_active[:][:, 0 : (300 - 10)])
y2 = np.mean(fr_silent[:][:, (300 - 10) : (600 - 20)])


for i in range(len(fr_silent)):
    ax.plot(
        2,
        np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        marker=".",
        color="darkgrey",
    )

for i in range(len(fr_active)):
    ax.plot(1, np.mean(fr_active[:][i, 0 : (300 - 10)]), marker=".", color="darkgrey")

for i in range(len(fr_active)):
    ax.plot(
        [1, 2],
        [
            np.mean(fr_active[:][i, 0 : (300 - 10)]),
            np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, y1, marker="o", color=colors[3])
ax.plot(2, y2, marker="o", color=colors[3])
ax.plot([1, 2], [y1, y2], "-", color=colors[3])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=fs - 1)
ax.set_yticklabels([""])
ax.set_title("v2", color=colors[3], fontsize=fs)

ax = fig.add_subplot(gs[2])
fr_silent = fr_silent_v3
fr_active = fr_active_v3

y1 = np.mean(fr_active[:][:, 0 : (300 - 10)])
y2 = np.mean(fr_silent[:][:, (300 - 10) : (600 - 20)])


for i in range(len(fr_silent)):
    ax.plot(
        2,
        np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        marker=".",
        color="darkgrey",
    )

for i in range(len(fr_active)):
    ax.plot(1, np.mean(fr_active[:][i, 0 : (300 - 10)]), marker=".", color="darkgrey")

for i in range(len(fr_active)):
    ax.plot(
        [1, 2],
        [
            np.mean(fr_active[:][i, 0 : (300 - 10)]),
            np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, y1, marker="o", color=colors[4])
ax.plot(2, y2, marker="o", color=colors[4])
ax.plot([1, 2], [y1, y2], "-", color=colors[4])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=fs - 1)
ax.set_yticklabels([""])
ax.set_title("v3", color=colors[4], fontsize=fs)

ax = fig.add_subplot(gs[3])
fr_silent = fr_silent_v4
fr_active = fr_active_v4

y1 = np.mean(fr_active[:][:, 0 : (300 - 10)])
y2 = np.mean(fr_silent[:][:, (300 - 10) : (600 - 20)])


for i in range(len(fr_silent)):
    ax.plot(
        2,
        np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        marker=".",
        color="darkgrey",
    )

for i in range(len(fr_active)):
    ax.plot(1, np.mean(fr_active[:][i, 0 : (300 - 10)]), marker=".", color="darkgrey")

for i in range(len(fr_active)):
    ax.plot(
        [1, 2],
        [
            np.mean(fr_active[:][i, 0 : (300 - 10)]),
            np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, y1, marker="o", color=colors[5])
ax.plot(2, y2, marker="o", color=colors[5])
ax.plot([1, 2], [y1, y2], "-", color=colors[5])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=fs - 1)
ax.set_yticklabels([""])
ax.set_title("v4", color=colors[5], fontsize=fs)

ax = fig.add_subplot(gs[4])
fr_silent = fr_silent_v5
fr_active = fr_active_v5

y1 = np.mean(fr_active[:][:, 0 : (300 - 10)])
y2 = np.mean(fr_silent[:][:, (300 - 10) : (600 - 20)])


for i in range(len(fr_silent)):
    ax.plot(
        2,
        np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        marker=".",
        color="darkgrey",
    )

for i in range(len(fr_active)):
    ax.plot(1, np.mean(fr_active[:][i, 0 : (300 - 10)]), marker=".", color="darkgrey")

for i in range(len(fr_active)):
    ax.plot(
        [1, 2],
        [
            np.mean(fr_active[:][i, 0 : (300 - 10)]),
            np.mean(fr_silent[:][i, (300 - 10) : (600 - 20)]),
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, y1, marker="o", color=colors[6])
ax.plot(2, y2, marker="o", color=colors[6])
ax.plot([1, 2], [y1, y2], "-", color=colors[6])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=fs - 1)
ax.set_yticklabels([""])
ax.set_title("v5", color=colors[6], fontsize=fs)

fig.suptitle("Hippocampal network", fontsize=fs)
plt.savefig("hippo_robust.png", dpi=500)