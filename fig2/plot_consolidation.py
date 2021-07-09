#%%
import numpy as np
import matplotlib.pyplot as plt

fr_active = np.load("data/consolidation/fr_active.npy")
fr_silent = np.load("data/consolidation/fr_silent.npy")

y1 = np.mean(fr_active, axis=0)
y2 = np.mean(fr_silent, axis=0)

y1err = np.std(fr_active, axis=0)
y2err = np.std(fr_silent, axis=0)

x = np.arange(0, 11, 1)

barlist = plt.bar(x - 0.2, y1, color="tab:green", width=0.4, yerr=y1err,edgecolor='black')
# barlist[0].set_hatch("/")
plt.bar(x + 0.2, y2, color="darkgrey", width=0.4, yerr=y2err,edgecolor='black')
plt.xticks(x)
ax = plt.axes()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ratio = 0.4
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
plt.ylabel("Firing rate (Hz)", fontsize=15)
plt.xlabel("Trial number", fontsize=15)
plt.savefig("consolidated.png", dpi=500)
plt.show()
