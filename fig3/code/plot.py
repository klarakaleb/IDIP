#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import seaborn as sns


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
N2_EX = 80


spikes = np.load("data/30_spikes.npy")
fr = get_network_firing_rates(spikes.T, 1000, len(TIME))
mean_e_fr = np.mean(fr[0:N2_EX], axis=0)
std_e_fr = np.std(fr[0:N2_EX], axis=0)
syn_w = np.load("data/30_w.npy")
syn_w = syn_w.reshape(len(syn_w), 1600)
# remove zeros
syn_w = syn_w.T[np.where(syn_w != 0)[1]]
mean_w = np.mean(syn_w, axis=0)
std_w = np.std(syn_w, axis=0)


e_input = np.load("../data/30_e_input.npy")
i_input = np.load("../data/30_i_input.npy")

e_input = e_input.T
i_input = i_input.T
spikes = spikes.T


isi_cv = []
for i in range(80):
    indices = np.where(spikes[i, -15001:-1001] == 1)  # last 10 seconds
    timings = TIME[indices]
    diffs = np.diff(timings)
    isi_cv.append(np.std(diffs) / np.mean(diffs))


#%%
####################################################################################################################################

fs = 13

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(4, 12, wspace=0.8, hspace=0.7)


ax = fig.add_subplot(gs[2, :2])
plt.hist(
    np.mean(fr[0:80, -15001:-1001], axis=1),
    edgecolor="black",
    color="white",
    bins=10,
    ec="k",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Firing rate (Hz)", fontsize=label)
ax.set_ylabel("Counts", fontsize=label)
plt.tick_params(labelsize=fs)
ax.set_xlim(0, 12)
ax.set_ylim(0, 40)
ax.set_xticks(np.arange(0, 12 + 1, 5))

ax = fig.add_subplot(gs[2, 2:4])
plt.hist(isi_cv, edgecolor="black", color="white", bins=7, ec="k")
plt.tick_params(labelsize=fs)
ax.set_xlabel("ISI CV", fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.xlim(0, 2)
ax.set_ylim(0, 40)
ax.set_yticks([])
ax.set_xticks(np.arange(0, 2 + 1, 1))


frs = np.load("conntarget/fr_mat.npy").mean(axis=2)

ax = fig.add_subplot(gs[3, :4])

connect = np.round(np.arange(0.5, 10.0 + 0.5, 0.5), 1)
targets = np.round(np.arange(4.0, 8.0 + 0.1, 0.1), 1)


target_labels = targets.tolist()
target_labels[:] = np.repeat(" ", len(target_labels))
target_labels[::5] = targets[::5][::-1]

connect_labels = connect[1:].tolist()
connect_labels[:] = np.repeat(" ", len(connect_labels))
connect_labels[::2] = connect[1:][::2] / 10

annot = np.array(
    [np.repeat(" ", len(connect_labels)) for i in range(len(target_labels))]
)
annot[25, 3] = "X"


sns.heatmap(
    np.flipud(frs[:, 1:]),
    cmap="viridis",
    vmin=0,
    yticklabels=target_labels,
    xticklabels=connect_labels,
    annot=annot,
    fmt="",
    ax=ax,
    # cbar_kws=dict(use_gridspec=False, location="bottom"),
)

ax.set_ylabel("Target input θ", fontsize=fs)
ax.set_xlabel("$\it{p}$ of inhibitory connections", fontsize=fs)
ax.tick_params(axis=u"both", which=u"both", length=0)
ax.tick_params(axis="both", labelsize=fs - 1)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fs - 1)
cbar.set_label("Firing rate (Hz)", fontsize=fs)


ax = fig.add_subplot(gs[0:2, 5:])

ax.plot(
    TIME[0:600000][::1000],
    mean_e_fr[0:600000][::1000],
    color="black",
    label="Firing rate",
)
ax.fill_between(
    TIME[0:600000][::1000],
    (mean_e_fr - std_e_fr)[0:600000][::1000],
    (mean_e_fr + std_e_fr)[0:600000][::1000],
    color="lightgray",
)
ax.spines["top"].set_visible(False)
ax.set_ylabel("Firing rate (Hz)", fontsize=fs)
ax.set_xlabel("Time (s)", fontsize=fs)
ax.tick_params(axis="y", labelsize=fs - 1)
ax.tick_params(axis="x", labelsize=fs - 1)


ax2 = ax.twinx()
ax2.plot(
    TIME[:600000][0::1000],
    mean_w[0:600],
    color="tab:blue",
    linewidth=2,
    label="Inhibitory weights",
)
ax2.fill_between(
    TIME[:600000][0::1000],
    mean_w[:600] - std_w[:600],
    mean_w[:600] + std_w[:600],
    color="lightsteelblue",
)
ax2.set_ylabel("Inhibitory weights", fontsize=label)
ax2.spines["right"].set_color("tab:blue")
ax2.yaxis.label.set_color("tab:blue")
ax2.tick_params(axis="y", colors="tab:blue", labelsize=fs - 1)
ax2.spines["top"].set_visible(False)
ax2.set_xlabel("Time (ms)", fontsize=fs)
ax2.set_ylim(0.07, 0.4)
ax.scatter(x=5.5, y=60, marker="X", color="black", ec="k", s=75, linewidth=0.5)
ax.scatter(x=50.5, y=60, marker="X", color="black", ec="k", s=75, linewidth=0.5)
ax.scatter(x=599.5, y=60, marker="X", color="black", ec="k", s=75, linewidth=0.5)
# plt.axvline(x=15, color="k", linestyle="--")


ax = fig.add_subplot(gs[2, 6:8])


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Neuron \nindex", fontsize=fs)
# ax.set_xlabel("Time (s)", fontsize=label)
ax.set_xticks(np.arange(0, 1 + 1, 1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = ""
labels[1] = "1s"
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(0, 80 + 1, 80))
ax.tick_params(axis="both", labelsize=fs - 1)

for i in range(80):
    ax.scatter(
        TIME[0:1000][np.where(spikes[:, 5000:6000][i] != 0)],
        np.repeat(i, len(TIME[5000:6000][np.where(spikes[:, 5000:6000][i] != 0)])),
        color="k",
        s=0.1,
    )

ax = fig.add_subplot(gs[2, 8:10])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.set_ylabel('Neuron index',labelpad = 5, fontsize=15)
ax.spines["bottom"].set_visible(False)
# ax.set_xlabel('Time (ms)',fontsize=15)
ax.set_xticks([])
ax.set_yticks([])

for i in range(80):
    ax.scatter(
        TIME[50000 : (50000 + 1000)][
            np.where(spikes[:, 50000 : (50000 + 1000)][i] != 0)
        ],
        np.repeat(
            i,
            len(
                TIME[50000 : (50000 + 1000)][
                    np.where(spikes[:, 50000 : (50000 + 1000)][i] != 0)
                ]
            ),
        ),
        color="k",
        s=0.1,
    )


ax = fig.add_subplot(gs[2, 10:12])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.set_xlabel('Time (ms)',fontsize=15)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_yticks(np.arange(0, 80+1, 80))


for i in range(80):
    ax.scatter(
        TIME[-1000:-1][np.where(spikes[:, -1000:-1][i] != 0)],
        np.repeat(i, len(TIME[-1000:-1][np.where(spikes[:, -1000:-1][i] != 0)])),
        color="k",
        s=0.1,
    )

ax = fig.add_subplot(gs[3, 6:8])
ax.set_ylim([-3 / 10000, 2 / 10000])
ax.plot(
    np.arange(0, N2_EX, 1),
    np.mean(e_input[0:N2_EX, 5000:6000], axis=1),
    "tab:green",
    label="excitatory",
)
ax.plot(
    np.arange(0, N2_EX, 1),
    -1 * np.mean(i_input[0:N2_EX, 5000:6000], axis=1),
    "tab:blue",
    label="inhbitory",
)
ax.plot(
    np.arange(0, N2_EX, 1),
    np.mean(e_input[0:N2_EX, 5000:6000], axis=1)
    - np.mean(i_input[0:N2_EX, 5000:6000], axis=1),
    "black",
    label="net",
)
ax.plot(np.arange(0, N2_EX, 1), np.repeat(0.0, N2_EX), "k--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Membrane \n current (nA)", fontsize=fs)
ax.set_xlabel("Neuron index", fontsize=fs)
ax.tick_params(axis="both", labelsize=fs - 1)
ax.set_xticks(np.arange(0, 80 + 1, 80))
ax.set_yticklabels([-0.25, -0.25, 0])


ax = fig.add_subplot(gs[3, 8:10])
ax.set_ylim([-3 / 10000, 2 / 10000])
ax.plot(
    np.arange(0, N2_EX, 1),
    np.mean(e_input[0:N2_EX, 50000:51000], axis=1),
    "tab:green",
    label="excitatory",
)
ax.plot(
    np.arange(0, N2_EX, 1),
    -1 * np.mean(i_input[0:N2_EX, 50000:51000], axis=1),
    "tab:blue",
    label="inhbitory",
)
ax.plot(
    np.arange(0, N2_EX, 1),
    np.mean(e_input[0:N2_EX, 50000:51000], axis=1)
    - np.mean(i_input[0:N2_EX, 50000:51000], axis=1),
    "black",
    label="net",
)
ax.plot(np.arange(0, N2_EX, 1), np.repeat(0.0, N2_EX), "k--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=fs - 1)
ax.spines["bottom"].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])


ax = fig.add_subplot(gs[3, 10:12])
ax.set_ylim([-3 / 10000, 2 / 10000])
ax.plot(
    np.arange(0, N2_EX, 1),
    np.mean(e_input[0:N2_EX, -1000:-1], axis=1),
    "tab:green",
    label="E",
)
ax.plot(
    np.arange(0, N2_EX, 1),
    -1 * np.mean(i_input[0:N2_EX, -1000:-1], axis=1),
    "tab:blue",
    label="I",
)
ax.plot(
    np.arange(0, N2_EX, 1),
    np.mean(e_input[0:N2_EX, -1000:-1], axis=1)
    - np.mean(i_input[0:N2_EX, -1000:-1], axis=1),
    "black",
    label="E - I",
)
ax.plot(np.arange(0, N2_EX, 1), np.repeat(0.0, N2_EX), "k--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# ax.set_xlabel('Neuron index',fontsize=15)
# ax.set_xticks(np.arange(0, 80+1, 80))
ax.set_xticks([])
ax.set_yticks([])
leg = ax.legend(ncol=3, bbox_to_anchor=(1.2, -0.1), handlelength=1)

plt.savefig("panel2.png", bbox_inches="tight", dpi=500)
