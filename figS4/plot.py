# %%
import numpy as np
import matplotlib.pyplot as plt
import pylustrator
from numba import jit
from brokenaxes import brokenaxes


@jit(nopython=True)
def get_network_firing_rates(spikes, timewindow, len_time):
    firing_rate = np.zeros(shape=(len(spikes), len_time))
    for i in range(0, len_time):
        if i + timewindow < len_time:
            firing_rate[:, i] = np.sum(spikes[:, i : i + timewindow], axis=1) / (
                timewindow / 1000
            )
    return firing_rate


idip_pre = np.load("data/fr_init.npy")
idip_pre = idip_pre.flatten()
idip_post = np.load("data/fr_end.npy")
idip_post = idip_post.flatten()

v_pre = np.load("data/fr_init_v.npy")
v_pre = v_pre.flatten()
v_post = np.load("data/fr_end_v.npy")
v_post = v_post.flatten()


hetero = np.load("data/v_hetero_repeats1.npy")
hetero2 = np.load("data/v_hetero_repeats2.npy")

mean_hetero = np.mean(hetero, axis=0)
std_hetero = np.std(hetero, axis=0)
mean_hetero2 = np.mean(hetero2, axis=0)
std_hetero2 = np.std(hetero2, axis=0)


cc_original = np.load("data/original_cc.npy")
cc_original_mean = np.mean(cc_original, axis=0)
cc_original_std = np.std(cc_original, axis=0)

cc_vogels = np.load("data/vogels_cc.npy")
cc_vogels_mean = np.mean(cc_vogels, axis=0)
cc_vogels_std = np.std(cc_vogels, axis=0)

cc_var1 = np.load("data/v1_cc.npy")
cc_var1_mean = np.mean(cc_var1, axis=0)
cc_var1_std = np.std(cc_var1, axis=0)

cc_var2 = np.load("data/v2_cc.npy")
cc_var2_mean = np.mean(cc_var2, axis=0)
cc_var2_std = np.std(cc_var2, axis=0)

cc_var3 = np.load("data/v3_cc.npy")
cc_var3_mean = np.mean(cc_var3, axis=0)
cc_var3_std = np.std(cc_var3, axis=0)

cc_var4 = np.load("data/v4_cc.npy")
cc_var4_mean = np.mean(cc_var4, axis=0)
cc_var4_std = np.std(cc_var4, axis=0)

cc_var5 = np.load("data/v5_cc.npy")
cc_var5_mean = np.mean(cc_var5, axis=0)
cc_var5_std = np.std(cc_var5, axis=0)

colors = [
    "black",
    "grey",
    "tab:green",
    "tab:orange",
    "tab:blue",
    "tab:purple",
    "tab:red",
]

spikes = np.load("data/30_spikes_memory_vogels.npy")
fr = get_network_firing_rates(spikes.T, 1000, len(spikes))


# %%
label = 12
tick = 11

pylustrator.start()

TIME = np.arange(0, 600, 0.001)


# fig, axs = plt.subplots(3, 3, figsize=(8.27, 11.69))
fig = plt.figure(figsize=(8.27, 11.69))
gs = fig.add_gridspec(4, 4)


ax = fig.add_subplot(gs[0, :2])
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
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


ax = fig.add_subplot(gs[1, 0])
ax.set_xlabel("Firing rate (Hz) \n before plasticity", fontsize=label)
ax.set_ylabel("Firing rate (Hz) \n after plasticity", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.set_ylim(0, 12)
ax.set_xlim(35, 60)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.scatter(idip_pre, idip_post, color="black", label="IDIP", alpha=0.25)

ax = fig.add_subplot(gs[1, 1])
ax.set_xlabel("Firing rate (Hz) \n before plasticity", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.set_ylim(0, 12)
ax.set_xlim(35, 60)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_yticks([])
ax.scatter(v_pre, v_post, color="silver", label="iSTDP", alpha=0.25)


ax = fig.add_subplot(gs[1, 2:4])

x = list(range(60)[::5])

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(-0.5, 8.5)
ax.plot(range(12), np.repeat(0, len(x)), "k--")

ax.errorbar(
    x=0,
    y=cc_original_mean[-1],
    yerr=cc_original_std[-1],
    fmt="o",
    color=colors[0],
    lw=3,
    label="IDIP",
)
ax.errorbar(
    x=2,
    y=cc_vogels_mean[-1],
    yerr=cc_vogels_std[-1],
    fmt="o",
    color=colors[1],
    lw=3,
    label="iSTDP",
)

ax.errorbar(
    x=4,
    y=cc_var1_mean[-1],
    yerr=cc_var1_std[-1],
    fmt="o",
    color=colors[2],
    lw=3,
    label="v1",
)
ax.errorbar(
    x=5,
    y=cc_var2_mean[-1],
    yerr=cc_var2_std[-1],
    fmt="o",
    color=colors[3],
    lw=3,
    label="v2",
)
ax.errorbar(
    x=6,
    y=cc_var3_mean[-1],
    yerr=cc_var3_std[-1],
    fmt="o",
    color=colors[4],
    lw=3,
    label="v3",
)
ax.errorbar(
    x=7,
    y=cc_var4_mean[-1],
    yerr=cc_var4_std[-1],
    fmt="o",
    color=colors[5],
    lw=3,
    label="v4",
)
ax.errorbar(
    x=8,
    y=cc_var5_mean[-1],
    yerr=cc_var5_std[-1],
    fmt="o",
    color=colors[6],
    lw=3,
    label="v5",
)

ax.xaxis.set_ticks(np.arange(0, 9, 1))
ax.set_xticklabels(
    ["IDIP", " ", "iSTDP", " ", "v1", "v2", "v3", "v4", "v5"], fontsize=label
)
ax.tick_params(axis=u"x", which=u"both", length=0)

ax.set_ylabel("Correlation coefficient", fontsize=label)
ax.tick_params(axis="y", labelsize=label)
ax.tick_params(axis="x", labelsize=label)

ax.set_ylim(-0.2, 1.05)
# ax.legend(ncol=8, fontsize=label, bbox_to_anchor=(2.25, -0.25))

colors = [
    "black",
    "black",
    "grey",
    "black",
    "tab:green",
    "tab:orange",
    "tab:blue",
    "tab:purple",
    "tab:red",
]

for color, tick_ in zip(colors, ax.xaxis.get_major_ticks()):
    tick_.label1.set_color(color)


TIME = np.arange(0, 1500, 0.001)

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


ysc = 0.7

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


ax = fig.add_subplot(gs[2, 0])
x1 = np.mean(fr_background[:, interval1], axis=1)
x2 = np.mean(fr_assembly[:, interval1], axis=1)
x_multi = [x1, x2]
ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    ec="k",
    bins=bins,
    linewidth=0.5,
)
ax.scatter(
    x=mean_fr[0, 0], y=ysc, marker="X", color="silver", ec="k", s=70, linewidth=0.5
)
ax.scatter(
    x=mean_fr[0, 1],
    y=ysc,
    marker="X",
    color="tab:orange",
    ec="k",
    s=0,
    linewidth=0.5,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.set_ylabel("Density", fontsize=12)
ax.set_xlabel("Firing Rate (Hz)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)


ax = fig.add_subplot(gs[2, 1])
x1 = np.mean(fr_background[:, interval2], axis=1)
x2 = np.mean(fr_assembly[:, interval2], axis=1)
x_multi = [x1, x2]
ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    ec="k",
    bins=bins,
    linewidth=0.5,
)
ax.scatter(
    x=mean_fr[1, 0], y=ysc, marker="X", color="silver", ec="k", s=75, linewidth=0.5
)
ax.scatter(
    x=mean_fr[1, 1], y=ysc, marker="X", color="tab:orange", ec="k", s=75, linewidth=0.5
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["left"].set_visible(False)
ax.set_yticks([])


ax = fig.add_subplot(gs[2, 2])
# x1 = np.mean(fr_background[:, 850000:860000], axis=1)
# x2 = np.mean(fr_assembly[:, 850000:860000], axis=1)
x1 = np.mean(fr_background[:, interval3], axis=1)
x2 = np.mean(fr_assembly[:, interval3], axis=1)
x_multi = [x1, x2]
ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    ec="k",
    bins=bins,
    linewidth=0.5,
)
ax.scatter(
    x=mean_fr[2, 0],
    y=ysc,
    marker="X",
    color="silver",
    ec="k",
    s=75,
    linewidth=0.5,
)
ax.scatter(
    x=mean_fr[2, 1],
    y=ysc,
    marker="X",
    color="tab:orange",
    ec="k",
    s=75,
    linewidth=0.5,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["left"].set_visible(False)
ax.set_yticks([])

ax = fig.add_subplot(gs[2, 3])
x1 = np.mean(fr_background[:, interval4], axis=1)
x2 = np.mean(fr_assembly[:, interval4], axis=1)
x_multi = [x1, x2]
ax.hist(
    x_multi,
    histtype="bar",
    density=True,
    color=["silver", "tab:orange"],
    ec="k",
    bins=bins,
    linewidth=0.5,
)

ax.scatter(
    x=mean_fr[3, 0], y=ysc, marker="X", color="silver", ec="k", s=75, linewidth=0.5
)
ax.scatter(
    x=mean_fr[3, 1], y=ysc, marker="X", color="tab:orange", ec="k", s=75, linewidth=0.5
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["left"].set_visible(False)
ax.set_yticks([])

range1 = np.r_[580000:820000]
range2 = np.r_[1180000:1220000]

bax = brokenaxes(xlims=((580, 820), (1180, 1220)), wspace=0.05, subplot_spec=gs[3, :4])
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


bax.set_xlabel("Time (s)", fontsize=label)
bax.set_ylabel("Firing rate (Hz)", fontsize=label)
bax.tick_params(axis="both", labelsize=tick)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl

plt.figure(1).axes[0].set_position([0.125000, 0.816381, 0.370652, 0.139162])
plt.figure(1).axes[1].set_position([0.125000, 0.636571, 0.168478, 0.118632])
plt.figure(1).axes[2].set_position([0.327174, 0.636571, 0.168478, 0.118632])
plt.figure(1).axes[3].set_position([0.654682, 0.636571, 0.298101, 0.318972])
plt.figure(1).axes[3].yaxis.labelpad = -5.440000
plt.figure(1).axes[4].set_position([0.125000, 0.422931, 0.168478, 0.118632])
plt.figure(1).axes[5].set_position([0.327174, 0.422931, 0.168478, 0.118632])
plt.figure(1).axes[6].set_position([0.548695, 0.422931, 0.168478, 0.118632])
plt.figure(1).axes[7].set_position([0.770216, 0.422931, 0.168478, 0.118632])
plt.figure(1).axes[8].set_position([0.125000, 0.180145, 0.648084, 0.167391])
plt.figure(1).axes[9].set_position([0.791986, 0.180145, 0.108014, 0.167391])
plt.figure(1).axes[10].set_position([0.125000, 0.180145, 0.775000, 0.167391])
#% end: automatic generated code from pylustrator

plt.savefig("S4grid.pdf")

# %%
