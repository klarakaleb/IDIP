#%%
import numpy as np
import matplotlib.pyplot as plt


a_mat = np.load("a_mat.npy")
mean_fr_mat = np.load("mean_fr_mat.npy")


fs = 13

fig = plt.figure(figsize=(5, 4))
gs = fig.add_gridspec(1, 1, wspace=0.5, hspace=0.8)

ax = fig.add_subplot(gs[0, 0])
plt.errorbar(range(5), a_mat.mean(axis=1), yerr=a_mat.std(axis=1), fmt="ok")
ax.set_xlabel("Target input", fontsize=fs)
ax.set_ylabel("# active CA1 cells", fontsize=fs)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks(np.arange(0, 4 + 1, 1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = np.arange(10, 35, 5)
ax.set_xticklabels(labels)
plt.tick_params(labelsize=fs - 1)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(20, 80)

plt.savefig("grid.png", dpi=500)
