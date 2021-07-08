#%%
import numpy as np
import matplotlib.pyplot as plt


cc_original = np.load("recurrent/fig3/data/cc.npy")
cc_original_mean = np.mean(cc_original, axis=0)
cc_original_std = np.std(cc_original, axis=0)

cc_vogels = np.load("recurrent/fig4/hetero/vogels/data/cc.npy")
cc_vogels_mean = np.mean(cc_vogels, axis=0)
cc_vogels_std = np.std(cc_vogels, axis=0)

cc_var1 = np.load("v1/data/cc.npy")
cc_var1_mean = np.mean(cc_var1, axis=0)
cc_var1_std = np.std(cc_var1, axis=0)

cc_var2 = np.load("v2/data/cc.npy")
cc_var2_mean = np.mean(cc_var2, axis=0)
cc_var2_std = np.std(cc_var2, axis=0)

cc_var3 = np.load("v3/data/cc.npy")
cc_var3_mean = np.mean(cc_var3, axis=0)
cc_var3_std = np.std(cc_var3, axis=0)

cc_var4 = np.load("v4/data/cc.npy")
cc_var4_mean = np.mean(cc_var4, axis=0)
cc_var4_std = np.std(cc_var4, axis=0)

cc_var5 = np.load("v5/data/cc.npy")
cc_var5_mean = np.mean(cc_var5, axis=0)
cc_var5_std = np.std(cc_var5, axis=0)


fs = 13

x = list(range(60)[::5])

colors = ['black','grey','tab:green','tab:orange','tab:blue','tab:purple','tab:red']

fig = plt.figure(figsize=(6, 4))
ax2 = plt.axes()

ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
# ax2.set_xlim(-0.5, 10.5)
ax2.set_xlim(-0.5, 8.5)
ax2.plot(range(12), np.repeat(0, len(x)), "k--")
# ax2.plot(x, np.repeat(1.0, len(x)), "k--")
"""
ax2.errorbar(
    x=0,
    y=cc_np_mean[-1],
    yerr=cc_np_std[-1],
    fmt="o",
    color="tab:olive",
    lw=3,
    label="NP",
)
"""

ax2.errorbar(
    x=0,
    y=cc_original_mean[-1],
    yerr=cc_original_std[-1],
    fmt="o",
    color=colors[0],
    lw=3,
    label="IDIP",
)
ax2.errorbar(
    x=2,
    y=cc_vogels_mean[-1],
    yerr=cc_vogels_std[-1],
    fmt="o",
    color=colors[1],
    lw=3,
    label="iSTDP",
)
ax2.errorbar(
    x=4,
    y=cc_var1_mean[-1],
    yerr=cc_var1_std[-1],
    fmt="o",
    color=colors[2],
    lw=3,
    label="v1",
)
ax2.errorbar(
    x=5,
    y=cc_var2_mean[-1],
    yerr=cc_var2_std[-1],
    fmt="o",
    color=colors[3],
    lw=3,
    label="v2",
)
ax2.errorbar(
    x=6,
    y=cc_var3_mean[-1],
    yerr=cc_var3_std[-1],
    fmt="o",
    color=colors[4],
    lw=3,
    label="v3",
)
ax2.errorbar(
    x=7,
    y=cc_var4_mean[-1],
    yerr=cc_var4_std[-1],
    fmt="o",
    color=colors[5],
    lw=3,
    label="v4",
)
ax2.errorbar(
    x=8,
    y=cc_var5_mean[-1],
    yerr=cc_var5_std[-1],
    fmt="o",
    color=colors[6],
    lw=3,
    label="v5",
)

ax2.xaxis.set_ticks(np.arange(0, 9, 1))
ax2.set_xticklabels(
    ["IDIP", " ", "iSTDP", " ", "v1", "v2", "v3", "v4", "v5"], fontsize=fs
)
ax2.tick_params(axis=u"x", which=u"both", length=0)

# Put a legend below current axis
"""
ax2.legend(
    loc="upper right",
    # fancybox=True,
    ncol=2,
    fontsize=11,
)
"""

ax2.set_ylabel("Correlation coefficient", fontsize=fs)
ax2.tick_params(axis="y", labelsize=fs)
ax2.tick_params(axis="x", labelsize=fs+1)

ax2.set_ylim(-0.2, 1.05)
# ax.legend(ncol=8, fontsize=fs, bbox_to_anchor=(2.25, -0.25))

colors = ['black','black','grey','black','tab:green','tab:orange','tab:blue','tab:purple','tab:red']

for color,tick in zip(colors,ax2.xaxis.get_major_ticks()):
    tick.label1.set_color(color) 


plt.tight_layout()
plt.savefig("cc.png", dpi=500)
