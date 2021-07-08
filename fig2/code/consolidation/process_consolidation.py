import numpy as np

dirs = [
    122,
    229,
    515,
    976,
    128,
    30,
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


fr_active = np.zeros(shape=(len(dirs), 11))
fr_silent = np.zeros(shape=(len(dirs), 11))
fr_inhibitory = np.zeros(shape=(len(dirs), 11))
fr_total = np.zeros(shape=(len(dirs), 11))



count = 0
for d in dirs:
    fr0 = np.load(str(d) + "56/0fr_trial.npy")
    active = set(np.where(fr0[:100, :] != 0)[0])
    silent = set(range(100)) - active
    active = np.array(list(active))
    silent = np.array(list(silent))
    fr_active[count, 0] = np.mean(fr0[active])
    fr_silent[count, 0] = np.mean(fr0[silent])
    fr_total[count, 0] = np.mean(fr0[10:110])
    count2 = 1
    for i in range(5, 55, 5):
        ii = i
        fr = np.load(str(d) + "56/" + str(ii) + "fr_trial.npy")
        fr_active[count, count2] = np.mean(fr[active])
        fr_silent[count, count2] = np.mean(fr[silent])
        fr_inhibitory[count, count2] = np.mean(fr[110])
        fr_total[count, count2] = np.mean(fr[10:110])
        count2 += 1
    count += 1


np.save("fr_silent.npy", fr_silent)
np.save("fr_active.npy", fr_active)
np.save("fr_inhibitory.npy", fr_inhibitory)
np.save("total.npy", fr_total)
