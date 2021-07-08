# %%

import numpy as np
import matplotlib.pyplot as plt

fr_ht = np.load("data/fr_ht.npy")
w_ht = np.load("data/w_ht.npy")
ii_ht = np.load("data/i_i_ht.npy")

fr_np = np.load("data/fr_np.npy")
w_np = np.load("data/w_np.npy")
ii_np = np.load("data/i_i_np.npy")


TIME = np.arange(0, 100, 0.001)


label = 12
tick = 11

fig = plt.figure(figsize=(6.5, 6))
gs = fig.add_gridspec(3, 3, wspace=1.0, hspace=1.0)


ax = fig.add_subplot(gs[0, 0])
ax.plot(TIME[0:-2001], fr_ht[0, 0:-2001], color="tab:green")
ax.plot(TIME[0:-2001], fr_ht[1, 0:-2001], color="tab:purple")
ax.set_ylim(0, 80)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks(np.arange(0, 80 + 1, 20))
# labels = [item.get_text() for item in ax.get_xticklabels()]
# ax.set_xticklabels(labels)

ax = fig.add_subplot(gs[0, 1])
# ax.plot(TIME[0:-2001], fr[2, 0:-2001], color="tab:orange")
# ax.plot(TIME[0:-2001], fr[3, 0:-2001], color="tab:blue")
ax.plot(TIME, ii_ht[:, 0], color="tab:orange")
ax.plot(TIME, ii_ht[:, 1], color="tab:blue")
ax.plot(TIME, np.repeat(0.35, len(TIME)), "--k")
# ax.set_ylim(0, 80)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel(r"S. Input ($\mu$S)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.set_yticks(np.arange(0, 0.41, 0.1))


ax = fig.add_subplot(gs[0, 2])
ax.plot(TIME[::1000], w_ht[:, 0, 0], color="tab:orange")
ax.plot(TIME[::1000], w_ht[:, 1, 1], color="tab:blue")
ax.set_ylim(-0.1, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("S. weights", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)

ax = fig.add_subplot(gs[1, 0])
ax.plot(TIME[0:-2001], fr_np[0, 0:-2001], color="tab:green")
ax.plot(TIME[0:-2001], fr_np[1, 0:-2001], color="tab:purple")
ax.set_ylim(0, 80)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks(np.arange(0, 80 + 1, 20))

ax = fig.add_subplot(gs[1, 1])
# ax.plot(TIME[0:-2001], fr[2, 0:-2001], color="tab:orange")
# ax.plot(TIME[0:-2001], fr[3, 0:-2001], color="tab:blue")
ax.plot(TIME, ii_np[:, 0], color="tab:orange")
ax.plot(TIME, ii_np[:, 1], color="tab:blue")
ax.plot(TIME, np.repeat(0.25, len(TIME)), "--k")
# ax.set_ylim(0, 80)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel(r"S. Input ($\mu$S)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.set_yticks(np.arange(0, 0.41, 0.1))


ax = fig.add_subplot(gs[1, 2])
ax.plot(TIME[::1000], w_np[:, 0, 0], color="tab:orange")
ax.plot(TIME[::1000], w_np[:, 1, 1], color="tab:blue")
ax.set_ylim(-0.1, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("S. weights", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)

plt.tight_layout()
plt.savefig("grid.png", dpi=500)


#%%

fig = plt.figure(figsize=(6.5, 4))
gs = fig.add_gridspec(2, 3, wspace=1.0, hspace=1.0)

ax = fig.add_subplot(gs[0, 0])
ax.plot(TIME[0:-2001], fr[0, 0:-2001], color="tab:green")
ax.plot(TIME[0:-2001], fr[1, 0:-2001], color="tab:purple")
ax.set_ylim(0, 80)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = fig.add_subplot(gs[0, 1])
ax.plot(TIME[0:-2001], fr[2, 0:-2001], color="tab:orange")
ax.plot(TIME[0:-2001], fr[3, 0:-2001], color="tab:blue")
ax.set_ylim(0, 80)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)

ax = fig.add_subplot(gs[0, 2])
ax.plot(TIME, syn_w[2], color="tab:orange")
ax.plot(TIME, syn_w[3], color="tab:blue")
ax.set_ylim(-0.1, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Synaptic weights", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)


ax = fig.add_subplot(gs[1, 0])
ax.plot(TIME[0:-2001], fr_noinh[0, 0:-2001], color="tab:green")
ax.plot(TIME[0:-2001], fr_noinh[1, 0:-2001], color="tab:purple")
ax.set_ylim(0, 80)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = fig.add_subplot(gs[1, 1])
ax.plot(TIME[0:-2001], fr_noinh[2, 0:-2001], color="tab:orange")
ax.plot(TIME[0:-2001], fr_noinh[3, 0:-2001], color="tab:blue")
ax.set_ylim(0, 80)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Firing Rate (Hz)", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)

ax = fig.add_subplot(gs[1, 2])
ax.plot(TIME, syn_w_noinh[2], color="tab:orange")
ax.plot(TIME, syn_w_noinh[3], color="tab:blue")
ax.set_ylim(-0.1, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Synaptic weights", fontsize=label)
ax.set_xlabel("Time (s)", fontsize=label)
ax.tick_params(axis="both", labelsize=tick)


plt.savefig("toy_model.png", bbox_inches="tight", dpi=500)
plt.show()
