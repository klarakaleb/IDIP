#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import pylustrator
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


TIME = np.arange(0, 1000, 0.001)
N2_EX = 80

label = 12
tick = 11

"""
spikes = np.load("data/original/30_spikes.npy")
fr = get_network_firing_rates(spikes.T, 1000, len(TIME))

np.save('fr.npy',fr)
"""

fr = np.load("fr.npy")

ee_spar = np.load("data/original/ee_spar.npy")

sd_var_mat = np.load("data/original/sd_var.npy")
sd_var_mat_v = np.load("data/vogels/var_sd.npy")

lr_var_mat = np.load("data/original/var_lr.npy")


#%%
"""
spikes_vogels = np.load("data/vogels/30_spikes.npy")
fr_vogels = get_network_firing_rates(spikes_vogels.T, 1000, len(TIME))


spikes_ii = np.load("data/II/30_spikes.npy")
fr_II = get_network_firing_rates(spikes_ii.T, 1000, len(TIME))

spikes_bigger = np.load("data/bigger/30_spikes.npy")
fr_bigger = get_network_firing_rates(spikes_bigger.T, 1000, len(TIME))
"""

mean_i_fr = np.mean(fr[N2_EX:], axis=0)
std_i_fr = np.std(fr[N2_EX:], axis=0)
syn_w = np.load("data/30_w.npy")
syn_w = syn_w.reshape(2000, 1600)
# remove zeros
syn_w = syn_w.T[np.where(syn_w[0, :] != 0)]
mean_w = np.mean(syn_w[-1])
std_w = np.std(syn_w[-1])

sds = np.load("data/original/sd_mat.npy").mean(axis=2)


label = 13
tick = 12

"""

spikes = spikes.T

isi_cv = []
for i in range(80, 100):
    indices = np.where(spikes[i, -15001:-1001] == 1)  # last 15 seconds
    timings = TIME[indices]
    diffs = np.diff(timings)
    isi_cv.append(np.std(diffs) / np.mean(diffs))

isi_cv = np.array(isi_cv)
np.save('isi_cv.npy',isi_cv)
"""
isi_cv = np.load("isi_cv.npy")

# %%
####################################################################################################################################
pylustrator.start()


# fig, axs = plt.subplots(3, 3, figsize=(8.27, 11.69))
fig = plt.figure(figsize=(8.27, 11.69))
gs = fig.add_gridspec(4, 3)


ax = fig.add_subplot(gs[0, 0])

ax.hist(
    np.mean(syn_w[:, -15001:-1001], axis=1),
    edgecolor="black",
    color="white",
    bins=10,
    ec="k",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Synaptic weights", fontsize=label)
ax.set_ylabel("Counts", fontsize=label)
ax.tick_params(labelsize=tick)


ax = fig.add_subplot(gs[0, 1])
ax.hist(
    np.mean(fr[80:100, -15001:-1001], axis=1),
    edgecolor="black",
    color="white",
    bins=10,
    ec="k",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Firing rate (Hz)", fontsize=label)
ax.set_ylabel("Counts", fontsize=label)
ax.tick_params(labelsize=tick)
ax.set_xlim(45, 60)
ax.set_ylim(0, 6)

ax = fig.add_subplot(gs[0, 2])


ax.hist(isi_cv, edgecolor="black", color="white", ec="k")
ax.tick_params(labelsize=tick)
ax.set_xlabel("ISI CV", fontsize=label)
ax.set_ylabel("Counts", fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 2)
ax.set_ylim(0, 6)


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


ax = fig.add_subplot(gs[1, 2])


fr_end_IDIP = np.load("data/original/fr_end.npy")
fr_end_IDIP = fr_end_IDIP.flatten()

fr_end_IDIP_mean = fr_end_IDIP.mean()
fr_end_IDIP_std = fr_end_IDIP.std()

fr_end_IDIP_II = np.load("data/II/fr_end.npy")
fr_end_IDIP_II = fr_end_IDIP_II.flatten()

fr_end_IDIP_II_mean = fr_end_IDIP_II.mean()
fr_end_IDIP_II_std = fr_end_IDIP_II.std()

fr_end_IDIP_bigger = np.load("data/bigger/fr_end.npy")
fr_end_IDIP_bigger = fr_end_IDIP_bigger.flatten()

fr_end_IDIP_bigger_mean = fr_end_IDIP_bigger.mean()
fr_end_IDIP_bigger_std = fr_end_IDIP_bigger.std()


fr_end_vogels = np.load("data/vogels/fr_end.npy")
fr_end_vogels = fr_end_vogels.flatten()

fr_end_vogels_mean = fr_end_vogels.mean()
fr_end_vogels_std = fr_end_vogels.std()

colors = ["black", "tab:olive", "tab:pink", "grey"]

ax.errorbar(
    x=0,
    y=fr_end_IDIP_mean,
    yerr=fr_end_IDIP_std,
    fmt="o",
    color=colors[0],
    lw=3,
)

ax.errorbar(
    x=1,
    y=fr_end_IDIP_II_mean,
    yerr=fr_end_IDIP_II_std,
    fmt="o",
    color=colors[1],
    lw=3,
)

ax.errorbar(
    x=2,
    y=fr_end_IDIP_bigger_mean,
    yerr=fr_end_IDIP_bigger_std,
    fmt="o",
    color=colors[2],
    lw=3,
)

ax.errorbar(
    x=3,
    y=fr_end_vogels_mean,
    yerr=fr_end_vogels_std,
    fmt="o",
    color=colors[3],
    lw=3,
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0, 12)
ax.set_ylabel("Firing rate (Hz)", fontsize=label)
plt.tick_params(labelsize=tick)
ax.set_xticks(np.arange(0, 3 + 1, 1))
ax.set_xticklabels([])


ax = fig.add_subplot(gs[2, 0])

ee_spar = ee_spar.reshape(11, 80 * 20)

ee_spar_std = np.std(ee_spar, axis=1)
ee_spar_mean = np.mean(ee_spar, axis=1)

ax.errorbar(range(11), ee_spar_mean, yerr=ee_spar_std, fmt="ok")
ax.set_ylim(0, 12)
ax.set_xlabel("E-E connectivity", fontsize=label)
ax.set_ylabel("Mean firing rate (Hz)", fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xticks(np.arange(0, 10 + 1, 2))
labels = np.arange(0, 11, 2) / 10
ax.set_xticklabels(labels)


plt.tick_params(labelsize=tick)


ax = fig.add_subplot(gs[2, 1])

sd_var_mat_std_std = np.std(np.std(sd_var_mat, axis=1), axis=1)
sd_var_mat_std = np.std(sd_var_mat.reshape(11, 80 * 20), axis=1)


sd_var_mat_v_std_std = np.std(np.std(sd_var_mat_v, axis=1), axis=1)
sd_var_mat_v_std = np.std(sd_var_mat_v.reshape(11, 80 * 20), axis=1)

ax.errorbar(range(11), sd_var_mat_std, yerr=sd_var_mat_std_std, fmt="ok")
ax.errorbar(
    range(11), sd_var_mat_v_std, yerr=sd_var_mat_v_std_std, fmt="o", color="grey"
)
ax.set_ylim(0.25, 2.5)
ax.set_xlabel("$\sigma_wrec$ ", fontsize=label)
ax.set_ylabel("Mean firing rate (Hz)", fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xticks(np.arange(0, 10 + 1, 2))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = np.arange(0, 11, 2) / 10
ax.set_xticklabels(labels)
plt.tick_params(labelsize=tick)

ax = fig.add_subplot(gs[2, 2])

lr_var_mat = lr_var_mat.reshape(4, 80 * 20)

lr_var_mat_std = np.std(lr_var_mat, axis=1)
lr_var_mat_mean = np.mean(lr_var_mat, axis=1)

ax.errorbar(range(4), lr_var_mat_mean, yerr=lr_var_mat_std, fmt="ok")
ax.set_ylim(0, 12)
ax.set_xlabel("$\eta$", fontsize=label)
ax.set_ylabel("Mean firing rate (Hz)", fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xticks(np.arange(0, 3 + 1, 1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = [0.5, 1, 5.0, 10.0]
ax.set_xticklabels(labels)
plt.tick_params(labelsize=tick)


ax = fig.add_subplot(gs[3, 0])


isi_cv = np.load("data/original/isi_cv.npy")
isi_cv = isi_cv.flatten()

isi_cv_mean = isi_cv.mean()
isi_cv_std = isi_cv.std()

isi_cv_II = np.load("data/II/isi_cv.npy")
isi_cv_II = isi_cv_II.flatten()

isi_cv_II_mean = isi_cv_II.mean()
isi_cv_II_std = isi_cv_II.std()

isi_cv_bigger = np.load("data/bigger/isi_cv.npy")
isi_cv_bigger = isi_cv_bigger.flatten()

isi_cv_bigger_mean = isi_cv_bigger.mean()
isi_cv_bigger_std = isi_cv_bigger.std()


isi_cv_vogels = np.load("data/vogels/isi_cv.npy")
isi_cv_vogels = isi_cv_vogels.flatten()

isi_cv_vogels_mean = isi_cv_vogels.mean()
isi_cv_vogels_std = isi_cv_vogels.std()


colors = ["black", "tab:olive", "tab:pink", "grey"]

ax.errorbar(
    x=0,
    y=isi_cv_mean,
    yerr=isi_cv_std,
    fmt="o",
    color=colors[0],
    lw=3,
)

ax.errorbar(
    x=1,
    y=isi_cv_II_mean,
    yerr=isi_cv_II_std,
    fmt="o",
    color=colors[1],
    lw=3,
)

ax.errorbar(
    x=2,
    y=isi_cv_bigger_mean,
    yerr=isi_cv_bigger_std,
    fmt="o",
    color=colors[2],
    lw=3,
)

ax.errorbar(
    x=3,
    y=isi_cv_mean,
    yerr=isi_cv_vogels_std,
    fmt="o",
    color=colors[3],
    lw=3,
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0, 2)
ax.set_ylabel("ISI CV", fontsize=label)
plt.tick_params(labelsize=tick)
ax.set_xticks(np.arange(0, 3 + 1, 1))


ax.set_xticklabels(
    ["IDIP", "$IDIP_{II}$", "$IDIP_{bigger}$", "iSTDP"], fontsize=label, rotation=45
)

for color, tick_ in zip(colors, ax.xaxis.get_major_ticks()):
    tick_.label1.set_color(color)


fr_ht = np.load("data/fr_ht.npy")
w_ht = np.load("data/w_ht.npy")
ii_ht = np.load("data/i_i_ht.npy")

fr_lt = np.load("data/fr_lt.npy")
w_lt = np.load("data/w_lt.npy")
ii_lt = np.load("data/i_i_lt.npy")

fr_np = np.load("data/fr_np.npy")
w_np = np.load("data/w_np.npy")
ii_np = np.load("data/i_i_np.npy")

TIME = np.arange(0, 100, 0.001)


label = 12
tick = 11

ax = fig.add_subplot(gs[3, 1])
ax.plot(TIME[0:-2001], fr_ht[0, 0:-2001], color="tab:green")
ax.plot(TIME[0:-2001], fr_ht[1, 0:-2001], color="tab:purple")
ax.set_ylim(20, 80)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks(np.arange(20, 80 + 1, 20))


ax = fig.add_subplot(gs[3, 2])
ax.plot(TIME[::1000], w_ht[:, 0, 0], color="tab:orange")
ax.plot(TIME[::1000], w_ht[:, 1, 1], color="tab:blue")
ax.set_ylim(-0.1, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("S. weights", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)


#% end: automatic generated code from pylustrator
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl

plt.figure(1).axes[0].set_position([0.095979, 0.862492, 0.121500, 0.091942])
plt.figure(1).axes[0].xaxis.labelpad = 2.400000
plt.figure(1).axes[0].get_xaxis().get_label().set_text("Synaptic \n  weights")
plt.figure(1).axes[1].set_position([0.299408, 0.862492, 0.121500, 0.091942])
plt.figure(1).axes[1].yaxis.labelpad = -4.403313
plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
plt.figure(1).axes[2].set_position([0.502838, 0.862492, 0.121500, 0.091942])
plt.figure(1).axes[2].yaxis.labelpad = -4.000000
plt.figure(1).axes[2].get_yaxis().get_label().set_text("")
plt.figure(1).axes[3].set_position([0.767861, 0.797479, 0.216243, 0.156955])
plt.figure(1).axes[4].set_position([0.428724, 0.369894, 0.213431, 0.167391])
plt.figure(1).axes[4].yaxis.labelpad = -4.000000
plt.figure(1).axes[4].get_yaxis().get_label().set_text("")
plt.figure(1).axes[5].set_position([0.095979, 0.369894, 0.207038, 0.167391])
plt.figure(1).axes[5].xaxis.labelpad = 3.823834
plt.figure(1).axes[5].yaxis.labelpad = 6.720000
plt.figure(1).axes[5].yaxis.labelpad = 4.800000
plt.figure(1).axes[5].get_xaxis().get_label().set_text("$\sigma_{wrec}$ ")
plt.figure(1).axes[5].get_yaxis().get_label().set_text("$\sigma$ firing rate")
plt.figure(1).axes[6].set_position([0.767861, 0.369894, 0.213431, 0.167391])
plt.figure(1).axes[6].xaxis.labelpad = 4.767667
plt.figure(1).axes[6].yaxis.labelpad = -4.000000
plt.figure(1).axes[6].get_yaxis().get_label().set_text("")
plt.figure(1).axes[7].set_position([0.767861, 0.605685, 0.216243, 0.156955])
plt.figure(1).axes[8].set_position([0.428724, 0.128360, 0.213431, 0.167391])
plt.figure(1).axes[9].set_position([0.767861, 0.134103, 0.213431, 0.167391])
plt.figure(1).axes[9].yaxis.labelpad = 2.400000
plt.figure(1).axes[9].get_yaxis().get_label().set_text("Synaptic weights")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_fontsize(12)
plt.figure(1).texts[0].set_position([0.340387, 0.405902])
plt.figure(1).texts[0].set_rotation(90.0)
plt.figure(1).texts[0].set_text("Firing rate (Hz)")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_fontsize(12)
plt.figure(1).texts[1].set_position([0.682588, 0.405902])
plt.figure(1).texts[1].set_rotation(90.0)
plt.figure(1).texts[1].set_text("Firing rate (Hz)")
#% end: automatic generated code from pylustrator

plt.savefig("S3_grid.pdf")
#%%

fig = plt.figure(figsize=((8.27 * 2 / 3) / 1.2, (11.69 * 1 / 4.5) / 1.2))
ax = plt.axes()


sns.heatmap(
    np.flipud(sds[:, 1:]),
    cmap="viridis",
    vmin=0,
    yticklabels=target_labels,
    xticklabels=connect_labels,
    annot=annot,
    fmt="",
    ax=ax,
)

# ax.imshow(np.flipud(sds[:, 1:]), cmap="viridis", vmin=0)

ax.set_ylabel("Target input θ", fontsize=label)
ax.set_xlabel("$\it{p}$ of inhibitory connections", fontsize=label)
ax.tick_params(axis=u"both", which=u"both", length=0)
ax.tick_params(axis="both", labelsize=tick)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=tick)
cbar.set_label("$\sigma$  firing rate", fontsize=label)

plt.tight_layout()
plt.savefig("heatmap.pdf")
