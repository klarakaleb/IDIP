import numpy as np

TIME = np.arange(0, 30000, 1)

dirs = [
    30,
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

fr_active = np.zeros(shape=(len(dirs), 3 * (30000 - 1000)))
fr_silent = np.zeros(shape=(len(dirs), 3 * (30000 - 1000)))
fr_inh = np.zeros(shape=(len(dirs), 3 * (30000 - 1000)))
weights = np.zeros(shape=(len(dirs), 3 * 30000))


for i in range(len(dirs)):
    print(i)
    dirname = str(dirs[i])
    print(dirname)
    fr0_i = np.load(dirname + "3_ni/0fr_trial.npy")
    fr1_i = np.load(dirname + "3_ni/1fr_trial.npy")
    fr2_i = np.load(dirname + "3_ni/fr_trial.npy")
    w0_i = np.load(dirname + "3_ni/0w_i_trial.npy")
    w1_i = np.load(dirname + "3_ni/1w_i_trial.npy")
    w2_i = np.load(dirname + "3_ni/w_i_trial.npy")
    # fr99 = np.genfromtxt(dirname+'/99fr_trial.csv',delimiter=',')
    active = set(np.where(fr0_i[:100, :] != 0)[0])
    active = np.array(list(active))
    silent = set(range(100)) - set(active)
    silent = np.array(list(silent))
    fr_i = np.concatenate(
        [fr0_i[:, :-1000], fr1_i[:, :-1000], fr2_i[:, :-1000]], axis=1
    )
    w_i = np.concatenate([w0_i[:], w1_i[:], w2_i[:]], axis=1)
    fr_silent[i, :] = np.mean(fr_i[silent], axis=0)
    fr_active[i, :] = np.mean(fr_i[active], axis=0)
    fr_inh[i, :] = fr_i[100]
    weights[i, :] = w_i[:,0,0]


np.save("fr_silent_ni.npy", fr_silent)
np.save("fr_active_ni.npy", fr_active)
np.save("fr_inh_ni.npy", fr_inh)
np.save("w_ni.npy", weights)

