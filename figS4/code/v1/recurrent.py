#%%

# IMPORT MODULES
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import random
import os
import sys
import datetime
from numba import jit


# SET SEED
# ----------------------------------------------------------------------------------------------------------------------

seed = sys.argv[1]
seed = float(seed)
seed = int(seed)

np.random.seed(seed)

print("seed: " + str(seed))

# PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# network
N1 = 100  # input layerq
N1E = 100  # of which excitatory
N2 = 100  # recurrent layer
N2E = 80  # of which excitatory
N = N1 + N2  # total number of neurons
C = 0.1  # connectivity
C_I = 2.5 * C
C_INPUT = 2 * C


# neuron model
TAU_M = 0.020  # membrane time constant[s]
V_REST = -0.060  # resting membrane potential [V]
R = 100  # * 1000000 * 1e6  # membrane resistance [Ohm]
THETA_M = -0.050  # spiking threshold [V]
T_REF = 0.002  # membrane refractory period [s]
V_E = 0  # excitatory reversal potential [V]
V_I = -0.080  # inhibitory reversal potential [V]

# synapses
GS = 1 / 1000  # 000000 * 1e6  # basic unit of synaptic conductance [S]
W = 1.0  # basic unit of synaptic weight
TAU_E = 0.005  # time constant for excitatory synapses [s]
TAU_I = 0.010  # time constant for inhibitory synapses[s]
SD = 0.1  # standard deviation for the weights sampling distribution
INPUT_COEFF = 2.5  # relative strenght of input synapses

# learning
W_MAX = 1.0  # maximum value for inhibitory synaptic weights
ETA_I = 0.1 * 40 / 1e6  # inhibitory learning rate
A_I = 1000  # inhibitory learning amplitude
THETA_I = -3.0 / 100  # 000000 * 1e6  # target input for inhibitory neurons
TAU_IL = 2 * 4 * TAU_M  # time constant for inhibitory learning [s]

# simulation
DT = 0.001  # simulation timestep [s]
T = 2000  # total time [s]
TIME = np.arange(
    0, T, DT
)  # np.linspace(start=0, stop=T, num=T * 1000)  # simulation time [s]
LEN_TIME = int(len(TIME))  # length of the simulation [ms]

# input spikes
FR = 10  # firing rate of inputs [Hz]
FRDT = FR * DT  # input spikes generator

# constants
EXP_M = np.exp(-DT / TAU_M)  # membrane exponential decay
EXP_IL = np.exp(-DT / TAU_IL)  # inhibitory learning exponential decay
EXP_I = np.exp(-DT / TAU_I)  # inhibitory synapses exponential decay
EXP_E = np.exp(-DT / TAU_E)  # excitatory synapses exponential decay


# DIRECTORY SETUP
# ============================================================================================================
path = "data/"

# UTILS (FUNCTIONS)
# ============================================================================================================
# generate random set of input spikes
def get_random_inputs(n, frdt, time):
    np.random.seed(seed)
    random_ns = np.random.rand(len(time), n).T  # due to the way the array is filled
    input_spikes = np.zeros(shape=(n, len(time)), dtype="int8")
    input_spikes[random_ns >= frdt] = 0
    input_spikes[random_ns < frdt] = 1
    return input_spikes


# generate a network
def create_network(n, n1, n1e, n2, n2e, seed, c, c_input, c_i):
    random.seed(seed)
    np.random.seed(seed)
    n_input_per_neuron = int(c_input * n1)
    n_e_per_neuron = int(c * n2e)
    n_e_per_i_neuron = int(c_i * n2e)
    n_i_per_e_neuron = int(c_i * (n2 - n2e))
    win = np.zeros((n1, n2), dtype=np.uint16)
    wrec = np.zeros((n2, n2), dtype=np.uint16)
    # to keep track of synapse types
    synapse_type_mat = np.zeros((n2, n2), dtype="int32")
    for i in range(0, n2):
        excitatory_inputs = np.random.choice(
            range(0, n1e), size=n_input_per_neuron, replace=False
        )
        if len(excitatory_inputs) != n_input_per_neuron:
            print("input connection error")
            break
        win[excitatory_inputs, i] = 1
        if i < n2e:
            excitatory_inputs = np.random.choice(
                range(0, n2e), size=n_e_per_neuron, replace=False
            )
            # avoid self connections
            SELF = False
            if i in excitatory_inputs:
                SELF = True
            while SELF:
                where = np.where(excitatory_inputs == i)
                # sample from the rest to avoid duplicate connections
                excitatory_inputs[where] = np.random.choice(
                    np.array([i for i in range(0, n2e) if i not in excitatory_inputs]),
                    size=len(where),
                    replace=False,
                )
                if i not in excitatory_inputs[where]:
                    SELF = False
            wrec[excitatory_inputs, i] = 1
            synapse_type_mat[excitatory_inputs, i] = 1
            inhibitory_inputs = np.random.choice(
                range(n2e, n2), size=n_i_per_e_neuron, replace=False
            )
            wrec[inhibitory_inputs, i] = 1
            synapse_type_mat[inhibitory_inputs, i] = -1
        if i >= n2e:
            excitatory_inputs = np.random.choice(
                range(0, n2e), size=n_e_per_i_neuron, replace=False
            )
            wrec[excitatory_inputs, i] = 1
            synapse_type_mat[excitatory_inputs, i] = 1

    return win, wrec, synapse_type_mat


@jit(nopython=True)
def get_network_firing_rates(spikes, timewindow, len_time):
    firing_rate = np.zeros(shape=(len(spikes), len_time))
    for i in range(0, len_time):
        if i + timewindow < len_time:
            firing_rate[:, i] = np.sum(spikes[:, i : i + timewindow], axis=1) / (
                timewindow / 1000
            )
    return firing_rate


# NETWORK CREATION
# ============================================================================================================

# keep synapse type mat just in case
w_in_mask, w_rec_mask, synapse_type_mat = create_network(
    N, N1, N1E, N2, N2E, seed, C, C_INPUT, C_I
)

# INPUT SPIKE GENERATOR - from D&A pg 30
# ============================================================================================================

input_spikes = get_random_inputs(N1, FRDT, TIME)
# input_spikes = np.load("../poisson_inputs/" + str(seed) + "_input_spikes.npy")


# RANDOM WEIGHTS
# ============================================================================================================

np.random.seed(seed)
n_syn = sum(sum(w_in_mask)) + sum(sum(w_rec_mask))
random_weights = np.random.lognormal(np.log(W), SD, n_syn)

# VARIABLES SETUP
# ============================================================================================================

# SYNAPSES

w_in = np.zeros(shape=(N2, N1))
w_in[w_in_mask.astype(bool)] = random_weights[: w_in_mask.sum()]
w_in = w_in.T  # from pre-post to post-pre
w_in = w_in * INPUT_COEFF


w_rec = np.zeros(shape=(N2, N2))
w_rec[w_rec_mask.astype(bool)] = random_weights[w_in_mask.sum() :]
w_rec = w_rec.T  # from pre-post to post-pre
w_rec[:N2E, N2E:] = w_rec[:N2E, N2E:] * 0.1  # initial inhibitory weights


w_in_mask = w_in_mask.T
w_rec_mask = w_rec_mask.T

# RECORDINGS

v = np.repeat(V_REST, N2)  #  initial voltage

#%%

# SIMULATION
# ============================================================================================================


@jit(nopython=True)
def run(v, w_rec):
    # INITIAL
    t_ref = np.zeros(shape=N2)
    i_i = np.zeros(shape=(N2 - N2E))
    s_t = np.zeros(shape=(N2))
    w_i = w_rec
    s = np.zeros(shape=(N2), dtype=np.int8)
    syn_c_input_i = np.zeros(shape=(N2, N1))
    syn_c_recurrent_i = np.zeros(shape=(N2, N2))
    factor = np.zeros(shape=(N2E, (N2 - N2E)))
    syn_input = np.zeros(shape=(N2))
    # outputs
    w = []
    spikes = []
    i_inputs = []
    for i in range(0, LEN_TIME, 1):
        # if i % (LEN_TIME / 10) == 0 and i != 0:
        #    print(i / LEN_TIME)
        t = TIME[i]
        # RECURRENT LAYER
        syn_input = (
            np.sum(syn_c_input_i, axis=1) * (v + V_E)
            + np.sum(syn_c_recurrent_i[:, :N2E], axis=1) * (v + V_E)
            - np.sum(syn_c_recurrent_i[:, N2E:], axis=1) * (v + V_I)
        )
        i_i *= EXP_IL
        i_i += syn_input[N2E:]
        # voltage and co update
        v[t_ref >= t] = V_REST  # refractory period
        dvdt = (V_REST - v) + R * -syn_input
        dvdt *= DT / TAU_M
        v[t_ref < t] += dvdt[t_ref < t]
        s[v >= THETA_M] = 1
        t_ref[v >= THETA_M] = t + T_REF
        s_t *= EXP_M + s
        # INPUT LAYER
        syn_c_input_i *= EXP_E
        syn_c_input_i += GS * w_in * input_spikes[:, i]
        # update synaptic conductances
        syn_c_recurrent_i[:, :N2E] *= EXP_E
        syn_c_recurrent_i[:, N2E:] *= EXP_I
        syn_c_recurrent_i += GS * w_i * s
        if t > 15:
            # INHIBITORY LEARNING
            dw = ETA_I * (-i_i + THETA_I) * s[N2E:]
            factor[:, dw > 0] = W_MAX - w_i[:N2E, N2E:][:, dw > 0]
            factor[:, dw < 0] = w_i[:N2E, N2E:][:, dw < 0]
            w_i[:N2E, N2E:] += A_I * factor * dw
            w_i *= w_rec_mask  #  prevent new connections from forming
        # RECORDINGS - copies otherwise rewriting
        if i % 1000 == 0:
            w.append(w_i[:N2E, N2E:].copy())
        i_inputs.append(i_i.copy())
        spikes.append(s.copy())
        # reset for next iteration
        s[:] = 0
        factor[:] = 0

    return w, spikes, i_inputs


tstart = datetime.datetime.now()

w, spikes, i_inputs = run(v, w_rec)


spikes = np.array(spikes)
w = np.array(w)

np.save(path + str(seed) + "_spikes.npy", spikes)
np.save(path + str(seed) + "_w.npy", w)


fr = get_network_firing_rates(spikes.T, 1000, LEN_TIME)

import matplotlib.pyplot as plt

plt.close("all")
fig = plt.figure()
ax = plt.axes()
ax.plot(TIME[::100], fr.transpose()[::100])
plt.savefig(path + str(seed) + "_fr.png")


i_inputs = np.array(i_inputs)
plt.close("all")
fig = plt.figure()
ax = plt.axes()
ax.plot(TIME[::100], i_inputs[::100])
ax.plot(TIME[::100], np.repeat(THETA_I, len(TIME[::100])), "k--")
plt.savefig(path + str(seed) + "_ii.png")

plt.close("all")
fig = plt.figure()
ax = plt.axes()
for i in range(N2 - N2E):
    ax.plot(TIME[::1000], w[:, :, i])
plt.savefig(path + str(seed) + "_w.png")


import scipy.stats

cc = np.zeros(shape=(133))
for i in range(133):
    cc[i] = scipy.stats.spearmanr(
        np.mean(fr[0:80, 0:15000], axis=1),
        np.mean(fr[0:80, (i * 15000) : ((i + 1) * 15000)], axis=1),
    )[0]

np.save(path + str(seed) + "_cc.npy", cc)

print(cc)

print("Mean ex fr: " + str(fr[:80, -16000:-1001].mean()))


tnow = datetime.datetime.now()
totalsecs = (tnow - tstart).total_seconds()

hrs = int(totalsecs // 3600)
mins = int(totalsecs % 3600) // 60
secs = int(totalsecs % 60)

timestamp = tnow.strftime("%b %d %Y %I:%M:%S %p").replace(" 0", " ")
print("{} ({} hrs {} mins {} secs elapsed)".format(timestamp, hrs, mins, secs))