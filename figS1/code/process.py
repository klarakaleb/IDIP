import numpy as np

seeds = [
    122,
    229,
    515,
    976,
    128,
    669,
    845,
    142,
    40,
    722,
    93,
    144,
    423,
    772,
    969,
    64,
    682,
    198,
    333,
]
targets = np.arange(10, 35, 5)

a_mat = np.zeros(shape=(len(targets), len(seeds)))
mean_fr_mat = np.zeros(shape=(len(targets), len(seeds)))

for ti, t in enumerate(targets):
    for si, s in enumerate(seeds):
        fr = np.load(str(s) + "100" + str(t) + "/fr_trial.npy")
        active = set(np.where(fr[:100, :] != 0)[0])
        active = np.array(list(active))
        a_mat[ti, si] = len(active)
        mean_fr_mat[ti, si] = fr[active].mean()


np.save("a_mat.npy", a_mat)
np.save("mean_fr_mat.npy", mean_fr_mat)


