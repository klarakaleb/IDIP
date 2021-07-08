#%%
import numpy as np
import matplotlib.pyplot as plt

import pylustrator

# pylustrator.start()


label = 12
tick = 11


fr_inh = np.load("data/fr_inh.npy")
w_inh = np.load("data/w.npy")
w_inh = w_inh[:, ::100]

y1 = np.mean(fr_inh, axis=0)
y1_err = np.std(fr_inh, axis=0)
x = np.arange(0, 70, 0.001)


w_inh = w_inh[:, np.r_[0:290, 300:590, 600:890]]

y2 = np.mean(w_inh, axis=0)
y2_err = np.std(w_inh, axis=0)

fig = plt.figure(figsize=(8.27, 11.69))
gs = fig.add_gridspec(4, 5)

ax = fig.add_subplot(gs[0, 0])
ax.set_ylim(0, 60)
ax.plot(x, y1[10000:80000], "tab:blue")
plt.fill_between(
    x,
    y1[10000:80000] - y1_err[10000:80000],
    y1[10000:80000] + y1_err[10000:80000],
    color="tab:blue",
    alpha=0.2,
)
ax.tick_params(axis="both", labelsize=tick)
ax.set_ylabel("Firing Rates (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.plot(np.repeat(19.0, 60), np.arange(0, 60, 1), "k--", alpha=0.2)
ax.plot(np.repeat(48.0, 60), np.arange(0, 60, 1), "k--", alpha=0.2)
# plt.fill_between(np.repeat(190.0,60),np.repeat(480.0,60),np.arange(0,60,1),color='red',alpha = 0.2)
ax.axvspan(19, 48, alpha=0.2, color="yellow")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


ax = fig.add_subplot(gs[1, 0])
ax.plot(x[0::100], y2[(100):(800)], "k--")
plt.fill_between(
    x[0::100],
    y2[(100):(800)] - y2_err[(100):(800)],
    y2[(100):(800)] + y2_err[(100):(800)],
    color="k",
    alpha=0.2,
)
ax.tick_params(axis="both", labelsize=tick)
ax.set_ylabel("Synaptic weights", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.plot(np.repeat(19.0, 50), np.arange(0, 0.05, 0.001), "k--", alpha=0.2)
ax.plot(np.repeat(48.0, 50), np.arange(0, 0.05, 0.001), "k--", alpha=0.2)
ax.set_ylim(0, 0.05)
ax.axvspan(19, 48, alpha=0.2, color="yellow")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


w_in = np.load("data/mean_w_in.npy")[:-1]


ax = fig.add_subplot(gs[2, 0])
ax.set_ylabel("Synaptic weights", fontsize=label)
ax.set_xlabel("# laps", fontsize=label)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


ax.plot(
    range(1, 51),
    w_in[:, 43, :],
    lw=2,
    color="grey",
)

ax.plot(
    range(1, 51),
    w_in[:, 44, :],
    lw=2,
    color="tab:green",
)


ax.set_title(r"$E \ → \ E$")
ax.tick_params(axis="both", labelsize=label)
for i in range(5, 51, 5):
    plt.axvline(x=i, color="k", linestyle="--")


ax = fig.add_subplot(gs[2, 1])

ax.plot(
    range(1, 51),
    w_in[:, 43, :].mean(axis=(1)),
    lw=3,
    color="grey",
)


ax.plot(
    range(1, 51),
    w_in[:, 44, :].mean(axis=(1)),
    lw=3,
    color="tab:green",
)

ax.set_title(r"$E \ → \ E$")
ax.tick_params(axis="both", labelsize=label)
for i in range(5, 51, 5):
    plt.axvline(x=i, color="k", linestyle="--")

ax.set_ylabel("Mean synaptic weights", fontsize=label)
ax.set_xlabel("# laps", fontsize=label)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


mat = np.load("data/mat.npy")

colors = [
    "black",
    "grey",
    "tab:green",
    "tab:orange",
    "tab:blue",
    "tab:purple",
    "tab:red",
]

ax = fig.add_subplot(gs[3, 0])


for i in range(20):
    ax.plot(
        2,
        mat[0, 1, i],
        marker=".",
        color="darkgrey",
    )

for i in range(20):
    ax.plot(1, mat[0, 0, i], marker=".", color="darkgrey")

for i in range(20):
    ax.plot(
        [1, 2],
        [
            mat[0, 0, i],
            mat[0, 1, i],
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, mat[0, 0].mean(), marker="o", color=colors[2])
ax.plot(2, mat[0, 1].mean(), marker="o", color=colors[2])
ax.plot([1, 2], [mat[0, 0].mean(), mat[0, 1].mean()], "-", color=colors[2])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label, labelpad=10)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=tick)
ax.set_title("v1", color=colors[2], fontsize=label)

ax = fig.add_subplot(gs[3, 1])

for i in range(20):
    ax.plot(
        2,
        mat[1, 1, i],
        marker=".",
        color="darkgrey",
    )

for i in range(20):
    ax.plot(1, mat[1, 0, i], marker=".", color="darkgrey")

for i in range(20):
    ax.plot(
        [1, 2],
        [
            mat[1, 0, i],
            mat[1, 1, i],
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, mat[1, 0].mean(), marker="o", color=colors[3])
ax.plot(2, mat[1, 1].mean(), marker="o", color=colors[3])
ax.plot([1, 2], [mat[1, 0].mean(), mat[1, 0].mean()], "-", color=colors[3])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=tick)
ax.set_yticklabels([""])
ax.set_title("v2", color=colors[3], fontsize=label)


ax = fig.add_subplot(gs[3, 2])


for i in range(20):
    ax.plot(
        2,
        mat[2, 1, i],
        marker=".",
        color="darkgrey",
    )

for i in range(20):
    ax.plot(1, mat[2, 0, i], marker=".", color="darkgrey")

for i in range(20):
    ax.plot(
        [1, 2],
        [
            mat[2, 0, i],
            mat[2, 1, i],
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, mat[2, 0].mean(), marker="o", color=colors[4])
ax.plot(2, mat[2, 1].mean(), marker="o", color=colors[4])
ax.plot([1, 2], [mat[2, 0].mean(), mat[2, 1].mean()], "-", color=colors[4])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=tick)
ax.set_yticklabels([""])
ax.set_title("v3", color=colors[4], fontsize=label)


ax = fig.add_subplot(gs[3, 3])


for i in range(20):
    ax.plot(
        2,
        mat[3, 1, i],
        marker=".",
        color="darkgrey",
    )

for i in range(20):
    ax.plot(1, mat[3, 0, i], marker=".", color="darkgrey")

for i in range(20):
    ax.plot(
        [1, 2],
        [
            mat[3, 0, i],
            mat[3, 1, i],
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, mat[3, 0].mean(), marker="o", color=colors[5])
ax.plot(2, mat[3, 1].mean(), marker="o", color=colors[5])
ax.plot([1, 2], [mat[3, 0].mean(), mat[3, 1].mean()], "-", color=colors[5])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=tick)
ax.set_yticklabels([""])
ax.set_title("v4", color=colors[5], fontsize=label)


ax = fig.add_subplot(gs[3, 4])


for i in range(20):
    ax.plot(
        2,
        mat[4, 1, i],
        marker=".",
        color="darkgrey",
    )

for i in range(20):
    ax.plot(1, mat[4, 0, i], marker=".", color="darkgrey")

for i in range(20):
    ax.plot(
        [1, 2],
        [
            mat[4, 0, i],
            mat[4, 1, i],
        ],
        "-",
        color="darkgrey",
        linewidth=0.5,
    )


ax.plot(1, mat[4, 0].mean(), marker="o", color=colors[6])
ax.plot(2, mat[4, 1].mean(), marker="o", color=colors[6])
ax.plot([1, 2], [mat[4, 0].mean(), mat[4, 1].mean()], "-", color=colors[6])

ax.set_ylim(0, 4.5)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["", ""], fontsize=label)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 1.0
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
ax.tick_params(axis="both", which="major", labelsize=tick)
ax.set_yticklabels([""])
ax.set_title("v5", color=colors[6], fontsize=label)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl

plt.figure(1).axes[0].set_position([0.086306, 0.807639, 0.367852, 0.110000])
plt.figure(1).axes[1].set_position([0.618349, 0.807639, 0.367852, 0.110000])
plt.figure(1).axes[2].set_position([0.086306, 0.415920, 0.367852, 0.110000])
plt.figure(1).axes[3].set_position([0.618349, 0.415920, 0.367852, 0.110000])
plt.figure(1).axes[3].yaxis.labelpad = 3.120000
plt.figure(1).axes[3].get_yaxis().get_label().set_text("Mean synaptic \n weights")
plt.figure(1).axes[4].set_position([0.086306, 0.616262, 0.142818, 0.101035])
plt.figure(1).axes[5].set_position([0.266601, 0.616262, 0.142818, 0.101035])
plt.figure(1).axes[6].set_position([0.446895, 0.616262, 0.142818, 0.101035])
plt.figure(1).axes[7].set_position([0.642630, 0.616262, 0.142818, 0.101035])
plt.figure(1).axes[8].set_position([0.843384, 0.616262, 0.142818, 0.101035])
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_fontsize(13)
plt.figure(1).texts[0].set_position([-0.029625, 0.940547])
plt.figure(1).texts[0].set_text("")
plt.figure(1).texts[0].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_fontsize(13)
plt.figure(1).texts[1].set_position([0.021161, 0.950813])
plt.figure(1).texts[1].set_text("A")
plt.figure(1).texts[1].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_fontsize(12)
plt.figure(1).texts[2].set_position([0.490326, 0.950813])
plt.figure(1).texts[2].set_text("B")
plt.figure(1).texts[2].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_fontsize(12)
plt.figure(1).texts[3].set_position([0.021161, 0.748075])
plt.figure(1).texts[3].set_text("C")
plt.figure(1).texts[3].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_fontsize(11)
plt.figure(1).texts[4].set_position([0.021161, 0.551326])
plt.figure(1).texts[4].set_text("D")
plt.figure(1).texts[4].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_fontsize(12)
plt.figure(1).texts[5].set_position([0.490326, 0.551326])
plt.figure(1).texts[5].set_text("E")
plt.figure(1).texts[5].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_color("#424242ff")
plt.figure(1).texts[6].set_fontsize(12)
plt.figure(1).texts[6].set_position([0.098549, 0.592387])
plt.figure(1).texts[6].set_text("OFF")
plt.figure(1).texts[6].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_color("#f0b357ff")
plt.figure(1).texts[7].set_fontsize(12)
plt.figure(1).texts[7].set_position([0.172310, 0.592387])
plt.figure(1).texts[7].set_text("ON")
plt.figure(1).texts[7].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_color("#4b4b4bff")
plt.figure(1).texts[8].set_fontsize(12)
plt.figure(1).texts[8].set_position([0.277509, 0.592387])
plt.figure(1).texts[8].set_text("OFF")
plt.figure(1).texts[8].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[9].new
plt.figure(1).texts[9].set_color("#f2bb62ff")
plt.figure(1).texts[9].set_fontsize(12)
plt.figure(1).texts[9].set_position([0.357316, 0.592387])
plt.figure(1).texts[9].set_text("ON")
plt.figure(1).texts[9].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[10].new
plt.figure(1).texts[10].set_color("#555555ff")
plt.figure(1).texts[10].set_fontsize(12)
plt.figure(1).texts[10].set_position([0.460097, 0.592387])
plt.figure(1).texts[10].set_text("OFF")
plt.figure(1).texts[10].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[11].new
plt.figure(1).texts[11].set_color("#f2bb62ff")
plt.figure(1).texts[11].set_fontsize(12)
plt.figure(1).texts[11].set_position([0.536276, 0.592387])
plt.figure(1).texts[11].set_text("ON")
plt.figure(1).texts[11].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[12].new
plt.figure(1).texts[12].set_color("#606060ff")
plt.figure(1).texts[12].set_fontsize(12)
plt.figure(1).texts[12].set_position([0.653567, 0.592387])
plt.figure(1).texts[12].set_text("OFF")
plt.figure(1).texts[12].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[13].new
plt.figure(1).texts[13].set_color("#f4c36dff")
plt.figure(1).texts[13].set_fontsize(12)
plt.figure(1).texts[13].set_position([0.730955, 0.592387])
plt.figure(1).texts[13].set_text("ON ")
plt.figure(1).texts[13].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[14].new
plt.figure(1).texts[14].set_color("#6b6b6bff")
plt.figure(1).texts[14].set_fontsize(12)
plt.figure(1).texts[14].set_position([0.854293, 0.592387])
plt.figure(1).texts[14].set_text("OFF")
plt.figure(1).texts[14].set_weight("bold")
plt.figure(1).text(
    0.5, 0.5, "New Text", transform=plt.figure(1).transFigure
)  # id=plt.figure(1).texts[15].new
plt.figure(1).texts[15].set_color("#f2bb62ff")
plt.figure(1).texts[15].set_fontsize(12)
plt.figure(1).texts[15].set_position([0.931681, 0.592387])
plt.figure(1).texts[15].set_text("ON")
plt.figure(1).texts[15].set_weight("bold")
#% end: automatic generated code from pylustrator
# plt.show()
plt.savefig("S2grid.pdf")

# %%
