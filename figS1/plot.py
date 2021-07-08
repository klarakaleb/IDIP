#%%
import numpy as np
import matplotlib.pyplot as plt

import pylustrator

# pylustrator.start()

#%%
# time duration of each lap [s]
TIME = np.linspace(start=0, stop=30, num=30 * 1000)
PLACE_FIELDS = np.linspace(start=2, stop=29, num=10).astype("int")
N1 = 10
LEN_TIME = len(TIME)
CURRENT_INPUT = 2 / 10000000000
A_PRE = 1
C_IC = 2 * ((6.54) ** 2)  # input current constant


pt = np.zeros(shape=(N1, LEN_TIME))  # scale for place tuning

for i in range(0, LEN_TIME, 1):
    d = PLACE_FIELDS - TIME[i]
    d[d > np.max(TIME) / 2] = np.max(TIME) - d[d > np.max(TIME) / 2]  # annular track
    d[d <= -np.max(TIME) / 2] = np.max(TIME) + d[d <= -np.max(TIME) / 2]
    pt[:, i] = A_PRE * np.exp(-((d ** 2) / C_IC))

input_current = pt * CURRENT_INPUT

fr = np.genfromtxt("data/0fr_trial.csv", delimiter=",")
label = 12
tick = 11

fig = plt.figure(figsize=(8.27, 11.69))
gs = fig.add_gridspec(5, 3)

# CURRENTS

ax = fig.add_subplot(gs[0, 0])
ax.set_ylabel("Current ( x100 pA)", fontsize=label)
ax.set_xlabel("Animal position (a.u.) ", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([])
for i in range(10):
    ax.plot(TIME, input_current[i])

# FR CA3
ax = fig.add_subplot(gs[1, 0])
ax.set_ylabel("Firing rate (Hz)", fontsize=label)
ax.set_xlabel("Animal position (a.u.) ", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([])
for i in range(10):
    ax.plot(TIME[0::100][0:-10], fr[i][0:-10])

w = np.load("data/w_in.npy")

# time duration of each lap [s]
TIME = np.linspace(start=0, stop=30, num=30 * 1000)


ax = fig.add_subplot(gs[0, 1])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("1")
ax.set_ylabel("Synaptic weight", fontsize=label)
ax.set_xlabel("Place Field", fontsize=label)
ax.set_ylim(-0.05, 1.5)
ax.set_xticks(np.arange(0, 10 + 1, 1))
ax.tick_params(axis="both", labelsize=tick)
for i in range(10):
    ax.scatter(i, w[:300, 43, i].mean(axis=0), marker="X")

ax = fig.add_subplot(gs[0, 2])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("2")
ax.set_ylabel("Synaptic weight", fontsize=label)
ax.set_xlabel("Place Field", fontsize=label)
ax.set_ylim(-0.05, 1.5)
ax.set_xticks(np.arange(0, 10 + 1, 1))
ax.tick_params(axis="both", labelsize=tick)
for i in range(10):
    ax.scatter(i, w[:300, 44, i].mean(axis=0), marker="X")

ax = fig.add_subplot(gs[1, 1])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("1")
ax.set_ylabel("Synaptic weight", fontsize=label)
ax.set_xlabel("Place Field", fontsize=label)
ax.set_ylim(-0.05, 1.5)
ax.set_xticks(np.arange(0, 10 + 1, 1))
ax.tick_params(axis="both", labelsize=tick)
for i in range(10):
    ax.scatter(i, w[-300:, 43, i].mean(axis=0), marker="X")


ax = fig.add_subplot(gs[1, 2])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("2")
ax.set_ylabel("Synaptic weight", fontsize=label)
ax.set_xlabel("Place Field", fontsize=label)
ax.set_ylim(-0.05, 1.5)
ax.set_xticks(np.arange(0, 10 + 1, 1))
ax.tick_params(axis="both", labelsize=tick)
for i in range(10):
    ax.scatter(i, w[-300:, 44, i].mean(axis=0), marker="X")


w_i = np.load("data/w_i.npy")[::10]
w_in = np.load("data/w_in.npy")[::10]

mean_w_i = np.load("data/mean_w_i.npy")
mean_w_in = np.load("data/mean_w_in.npy")

fr99 = np.load("data/99fr_trial.npy")

inputs = np.load("data/inputs.npy")
mean_inputs = np.load("data/inputs_mean.npy")

active = set(np.where(fr99[:100, :] != 0)[0])
silent = set(range(0, 100)) - active

active = np.array(list(active))
silent = np.array(list(silent))


ax = fig.add_subplot(gs[2, 0])
ax.plot(
    range(len(w_i)),
    w_in[:, active, :].reshape(
        w_in[:, active, :].shape[0],
        w_in[:, active, :].shape[1] * w_in[:, active, :].shape[2],
    ),
    alpha=0.25,
    lw=0.25,
    color="tab:green",
)
ax.plot(
    range(len(w_i)),
    w_in[:, silent, :].reshape(
        w_in[:, silent, :].shape[0],
        w_in[:, silent, :].shape[1] * w_in[:, silent, :].shape[2],
    ),
    alpha=0.25,
    lw=0.25,
    color="grey",
)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylabel("Synaptic weights", fontsize=label)
ax.set_xlabel("# laps", fontsize=label)
ax.set_title(r"$E \ → \ E$")
ax.tick_params(axis="both", labelsize=label)
ax.set_xticks(np.arange(0, 3000 + 1, 600))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = np.arange(0, 100 + 1, 20)
ax.set_xticklabels(labels)

ax = fig.add_subplot(gs[2, 1])
ax.set_ylabel("Synaptic weights", fontsize=label)
ax.set_xlabel("# laps", fontsize=label)
ax.plot(range(900), w_in[:900, active, :].mean(axis=(1)), color="tab:green", lw=0.25)
ax.plot(range(900), w_in[:900, active, :].mean(axis=(1, 2)), color="tab:green", lw=2)
ax.plot(range(900), w_in[:900, silent, :].mean(axis=(1)), color="grey", lw=0.25)
ax.plot(range(900), w_in[:900, silent, :].mean(axis=(1, 2)), color="grey")
ax.set_title(r"$E \ → \ E$")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="both", labelsize=label)
ax.set_xticks(np.arange(0, 900 + 1, 300))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = np.arange(0, 30 + 1, 10)
ax.set_xticklabels(labels)


ax = fig.add_subplot(gs[3, 0])
ax.plot(range(900), w_i[:900, 0, 0], color="tab:blue", alpha=0.5, lw=0.25)
ax.plot(
    range(900)[::30],
    mean_w_i[:30, 0, 0],
    color="tab:blue",
    linewidth=2,
)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="both", labelsize=label)
ax.set_ylabel("Synaptic weights", fontsize=label)
ax.set_xlabel("# laps", fontsize=label)
ax.set_xticks(np.arange(0, 900 + 1, 300))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = np.arange(0, 30 + 1, 10)
ax.set_xticklabels(labels)
ax.set_title(r"$I \ → \ E$")

ax = fig.add_subplot(gs[3, 1])
ax.plot(range(900), inputs[:9000][::10], color="black", alpha=0.5, lw=0.25)
ax.plot(
    range(900)[::30],
    mean_inputs[:30],
    color="black",
    linewidth=2,
)
ax.plot(range(900), np.repeat(0.2, 900), ":", color="tab:red", lw=3.0)


ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="both", labelsize=label)
ax.set_ylabel(r"Synaptic input ($\mu$S)", fontsize=label)
ax.set_xlabel("# laps", fontsize=label)
ax.set_xticks(np.arange(0, 900 + 1, 300))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = np.arange(0, 30 + 1, 10)
ax.set_xticklabels(labels)

fr99 = np.load("data/99fr_trial.npy")
fr0 = np.load("data/0fr_trial.npy")
#%%
# time duration of each lap [s]
TIME = np.linspace(start=0, stop=30, num=30 * 1000)

ax = fig.add_subplot(gs[4, 0])
ax.set_ylabel("Firing rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(20, 75)
ax.plot(TIME[0::100][0:-10], fr0[100][0:-10], color="grey", lw=2.0, label="lap 1")
ax.plot(TIME[0::100][0:-10], fr99[100][0:-10], color="black", lw=2.0, label="lap 100")
ax.legend()


a_mat = np.load("data/a_mat.npy")
mean_fr_mat = np.load("data/mean_fr_mat.npy")


ax = fig.add_subplot(gs[4, 1])
plt.errorbar(range(5), a_mat.mean(axis=1), yerr=a_mat.std(axis=1), fmt="ok")
ax.set_xlabel("Target input", fontsize=label)
ax.set_ylabel("# active CA1 cells", fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks(np.arange(0, 4 + 1, 1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = np.arange(10, 35, 5)
ax.set_xticklabels(labels)
plt.tick_params(labelsize=tick)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(20, 80)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl

plt.figure(1).axes[0].set_position([0.095212, 0.825121, 0.238029, 0.121242])
plt.figure(1).axes[0].yaxis.labelpad = -1.092302
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[1].set_position([0.095212, 0.639890, 0.238029, 0.121242])
plt.figure(1).axes[2].set_position([0.452256, 0.825121, 0.171381, 0.121242])
plt.figure(1).axes[2].title.set_position([0.422170, 1.000000])
plt.figure(1).axes[2].title.set_text("")
plt.figure(1).axes[2].xaxis.labelpad = -4.000000
plt.figure(1).axes[2].get_xaxis().get_label().set_text("")
plt.figure(1).axes[3].set_position([0.737891, 0.825121, 0.171381, 0.121242])
plt.figure(1).axes[3].title.set_position([0.500000, 0.992944])
plt.figure(1).axes[3].title.set_text("")
plt.figure(1).axes[3].xaxis.labelpad = -4.000000
plt.figure(1).axes[3].yaxis.labelpad = -4.000000
plt.figure(1).axes[3].get_xaxis().get_label().set_text("")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("")
plt.figure(1).axes[4].set_position([0.452256, 0.639890, 0.171381, 0.121242])
plt.figure(1).axes[4].title.set_position([0.500000, 1.000000])
plt.figure(1).axes[4].title.set_text("")
plt.figure(1).axes[5].set_position([0.737891, 0.639890, 0.171381, 0.121242])
plt.figure(1).axes[5].title.set_position([0.500000, 1.007056])
plt.figure(1).axes[5].title.set_text("")
plt.figure(1).axes[5].yaxis.labelpad = -12.000000
plt.figure(1).axes[5].get_yaxis().get_label().set_text("")
plt.figure(1).axes[6].set_position([0.095212, 0.404141, 0.214227, 0.134714])
plt.figure(1).axes[6].get_yaxis().get_label().set_weight("normal")
plt.figure(1).axes[7].set_position([0.409411, 0.404141, 0.214227, 0.134714])
plt.figure(1).axes[7].yaxis.labelpad = -4.000000
plt.figure(1).axes[7].get_yaxis().get_label().set_text("")
plt.figure(1).axes[8].set_position([0.737891, 0.404141, 0.214227, 0.134714])
plt.figure(1).axes[8].yaxis.labelpad = 194.929271
plt.figure(1).axes[8].get_yaxis().get_label().set_text("")
plt.figure(1).axes[9].set_position([0.095212, 0.168392, 0.214227, 0.134714])
plt.figure(1).axes[10].set_position([0.409411, 0.168392, 0.214227, 0.134714])
plt.figure(1).axes[11].set_position([0.737891, 0.168392, 0.214227, 0.134714])
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_fontsize(12)
plt.figure(1).texts[0].set_position([0.029625, 0.243370])
plt.figure(1).texts[0].set_rotation(90.0)
plt.figure(1).texts[0].set_text("")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_fontname("DejaVu Sans")
plt.figure(1).texts[1].set_fontsize(12)
plt.figure(1).texts[1].set_position([0.013906, 0.952524])
plt.figure(1).texts[1].set_text("A")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_fontsize(12)
plt.figure(1).texts[2].set_position([0.013906, 0.774594])
plt.figure(1).texts[2].set_text("B")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_fontsize(12)
plt.figure(1).texts[3].set_position([0.351270, 0.952524])
plt.figure(1).texts[3].set_text("C")
plt.figure(1).texts[3].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_fontsize(12)
plt.figure(1).texts[4].set_position([0.351270, 0.774594])
plt.figure(1).texts[4].set_text("D")
plt.figure(1).texts[4].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_fontsize(12)
plt.figure(1).texts[5].set_position([0.645103, 0.952524])
plt.figure(1).texts[5].set_text("E ")
plt.figure(1).texts[5].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_fontsize(12)
plt.figure(1).texts[6].set_position([0.645103, 0.774594])
plt.figure(1).texts[6].set_text("F")
plt.figure(1).texts[6].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_fontsize(12)
plt.figure(1).texts[7].set_position([0.013906, 0.562447])
plt.figure(1).texts[7].set_text("G")
plt.figure(1).texts[7].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_fontsize(12)
plt.figure(1).texts[8].set_position([0.351270, 0.562447])
plt.figure(1).texts[8].set_text("H")
plt.figure(1).texts[8].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[9].new
plt.figure(1).texts[9].set_fontsize(12)
plt.figure(1).texts[9].set_position([0.645103, 0.562447])
plt.figure(1).texts[9].set_text("I")
plt.figure(1).texts[9].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[10].new
plt.figure(1).texts[10].set_fontsize(12)
plt.figure(1).texts[10].set_position([0.013906, 0.328914])
plt.figure(1).texts[10].set_text("J")
plt.figure(1).texts[10].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[11].new
plt.figure(1).texts[11].set_fontsize(12)
plt.figure(1).texts[11].set_position([0.351270, 0.328914])
plt.figure(1).texts[11].set_text("K")
plt.figure(1).texts[11].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[12].new
plt.figure(1).texts[12].set_fontsize(12)
plt.figure(1).texts[12].set_position([0.645103, 0.328914])
plt.figure(1).texts[12].set_text("L")
plt.figure(1).texts[12].set_weight("bold")
#% end: automatic generated code from pylustrator
# plt.show()
plt.savefig("S1_grid.pdf")
