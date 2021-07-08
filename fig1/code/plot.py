#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_binned_current(current, window, time):
    c = np.zeros(shape=(len(current), int(len(time))))
    for i in range(0, len(time)):
        if i + window < len(current[0]):
            c[:, i] = np.sum(current[:, i : i + window], axis=1) / (window)
    return c


e_input2 = np.load("30100/0e_input.npy")
i_input2 = np.load("30100/0i_input.npy")
TIME = np.arange(0, 30, 0.001)


c0 = get_binned_current(e_input2, 1000, TIME) * 0.001
ci0 = get_binned_current(i_input2, 1000, TIME) * 0.001


lw = 2
mks = 3
x = range(10)

fs = 13

fig = plt.figure(figsize=(5, 4))
gs = fig.add_gridspec(2, 2, wspace=0.2, hspace=0.4)


# currents

# currents

i = 43
inh = ci0[:100][0:-1000]
ex = c0[:100][0:-1000]
net = c0[:100][0:-1000] + (-1) * ci0[:100][0:-1000]+ 10][0:-1000]

ax = fig.add_subplot(gs[0, 0])
ax.plot(TIME[0:-1000], -1 * inh, color="tab:blue", linewidth=lw)
ax.plot(TIME[0:-1000], ex, color="purple", alpha=1, linewidth=lw)
ax.plot(TIME[0:-1000], net, color="tab:green", linewidth=lw)

ax.plot(TIME[0:-1000], np.repeat(0, len(TIME) - 1000), "k--", alpha=0.5)
ax.plot(TIME[0:-1000], np.repeat(np.min(net), len(TIME) - 1000), "k--")
ax.plot(np.repeat(14, 10), np.linspace(-10, 5, 10), "k--", alpha=0.5)
ax.set_ylabel("Membrane \n current (nA)", fontsize=fs)
ax.set_xlabel("Animal \n position (a.u.)", labelpad=15, fontsize=fs)
ax.tick_params(labelsize=fs - 1)
ax.set_title("1")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(-8 / 100, 3.5 / 100)
ax.set_xticks([])


i = 44
inh = ci0[i + 10][0:-1000]
ex = c0[i + 10][0:-1000]
net = c0[i + 10][0:-1000] + (-1) * ci0[i + 10][0:-1000]


ax = fig.add_subplot(gs[0, 1])
ax.plot(TIME[0:-1000], -1 * inh, color="tab:blue", linewidth=lw)
ax.plot(TIME[0:-1000], ex, color="purple", alpha=1, linewidth=lw)
ax.plot(TIME[0:-1000], net, color="tab:green", linewidth=lw)


ax.plot(TIME[0:-1000], np.repeat(0, len(TIME) - 1000), "k--", alpha=0.5)
ax.plot(TIME[0:-1000], np.repeat(np.min(net), len(TIME) - 1000), "k--")
ax.plot(np.repeat(14, 10), np.linspace(-10, 5, 10), "k--", alpha=0.5)
# ax.set_ylabel('Membrane currents',labelpad=30,fontsize=20)
ax.set_xlabel("Animal \n position (a.u.)", labelpad=15, fontsize=fs)
ax.tick_params(labelsize=fs - 1)
ax.set_title("2")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_ylim(-8 / 100, 3.5 / 100)
ax.set_xticks([])
# ax.set_yticks([])
ax.set_yticklabels([])

# plt.savefig('current_prototype.pdf',bbox_inches='tight',dpi=500)
plt.savefig("current0.png", bbox_inches="tight", dpi=500)
########################################################################################


e_input2 = np.genfromtxt("30100/99e_input.txt")
i_input2 = np.genfromtxt("30100/99i_input.txt")
TIME = np.arange(0, 30, 0.001)


c0 = get_binned_current(e_input2, 1000, TIME) * 0.001
ci0 = get_binned_current(i_input2, 1000, TIME) * 0.001


fig = plt.figure(figsize=(5, 4))
gs = fig.add_gridspec(2, 2, wspace=0.2, hspace=0.4)


# currents

i = 43
inh = ci0[:100][0:-1000]
ex = c0[:100][0:-1000]
net = c0[:100][0:-1000] + (-1) * ci0[:100][0:-1000]

ax = fig.add_subplot(gs[0, 0])
ax.plot(TIME[0:-1000], -1 * inh, color="tab:blue", linewidth=lw)
ax.plot(TIME[0:-1000], ex, color="purple", alpha=1, linewidth=lw)
ax.plot(TIME[0:-1000], net, color="tab:green", linewidth=lw)

ax.plot(TIME[0:-1000], np.repeat(0, len(TIME) - 1000), "k--", alpha=0.5)
ax.plot(TIME[0:-1000], np.repeat(np.min(net), len(TIME) - 1000), "k--")
ax.plot(np.repeat(14, 10), np.linspace(-10, 5, 10), "k--", alpha=0.5)
ax.set_ylabel("Membrane \n current (nA)", fontsize=fs)
ax.set_xlabel("Animal \n position (a.u.)", labelpad=15, fontsize=fs)
ax.tick_params(labelsize=fs - 1)
ax.set_title("1")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(-8 / 100, 3.5 / 100)
ax.set_xticks([])


i = 44
inh = ci0[i + 10][0:-1000]
ex = c0[i + 10][0:-1000]
net = c0[i + 10][0:-1000] + (-1) * ci0[i + 10][0:-1000]


ax = fig.add_subplot(gs[0, 1])
ax.plot(TIME[0:-1000], -1 * inh, color="tab:blue", linewidth=lw)
ax.plot(TIME[0:-1000], ex, color="purple", alpha=1, linewidth=lw)
ax.plot(TIME[0:-1000], net, color="tab:green", linewidth=lw)


ax.plot(TIME[0:-1000], np.repeat(0, len(TIME) - 1000), "k--", alpha=0.5)
ax.plot(TIME[0:-1000], np.repeat(np.min(net), len(TIME) - 1000), "k--")
ax.plot(np.repeat(14, 10), np.linspace(-10, 5, 10), "k--", alpha=0.5)
# ax.set_ylabel('Membrane currents',labelpad=30,fontsize=20)
ax.set_xlabel("Animal \n position (a.u.)", labelpad=15, fontsize=fs)
ax.tick_params(labelsize=fs - 1)
ax.set_title("2")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_ylim(-8 / 100, 3.5 / 100)
ax.set_xticks([])
# ax.set_yticks([])
ax.set_yticklabels([])


plt.savefig("current99.png", bbox_inches="tight", dpi=500)


###########################################################################


fr0 = np.load("30100/0fr_trial.npy", delimiter=",")
fr99 = np.load("30100/99fr_trial.npy")

fig = plt.figure(figsize=(10, 2))
gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0.4)


ax = fig.add_subplot(gs[0, 0])
sns.heatmap(
    fr0[10:110, 0:-11], cmap="viridis", vmax=20, cbar_kws={"label": "Firing Rates (Hz)"}
)
# ax.set_ylabel('Neuron index',labelpad=30,fontsize=15)
cax = plt.gcf().axes[-1]
ax.set_xlabel("Animal position (a.u.)", labelpad=15, fontsize=fs)
ax.set_ylabel("CA1 neuron index", fontsize=fs, labelpad=30)
ax.tick_params(axis="both", labelsize=fs - 1)
# ax.set_yticks(np.arange(1, 100, 1))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_yticks([])
ax.set_xticks([])


ax = fig.add_subplot(gs[0, 1])
sns.heatmap(fr99[10:110, 0:-11], cmap="viridis", vmax=20)

cax = plt.gcf().axes[-1]
ax.set_xlabel("Animal position (a.u.)", labelpad=15, fontsize=fs)
ax.tick_params(axis="both", labelsize=fs - 1)
# ax.set_yticks(np.arange(0, 100+1, 100))
ax.set_xticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fs)
cbar.set_label("Firing rate (Hz)", fontsize=fs)
ax.set_yticks([])
ax.set_xticks([])


plt.tick_params(axis="both", labelsize=0)


plt.tight_layout()

plt.savefig("fr_heatmap.png", dpi=500, bbox_inches="tight")

# %%

