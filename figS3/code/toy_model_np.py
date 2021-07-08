# %%
import numpy as np
import random
import os
import sys
import datetime
from numba import jit
import matplotlib.pyplot as plt


# SET SEED
# ----------------------------------------------------------------------------------------------------------------------

seed = 30
np.random.seed(seed)

print("seed: " + str(seed))

# PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# network
N = 4


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
ETA_I = 0  # inhibitory learning rate
A_I = 1000  # inhibitory learning amplitude
THETA_I = 3.5 / 10  # 000000 * 1e6  # target input for inhibitory neurons
TAU_IL = 2 * 4 * TAU_M  # time constant for inhibitory learning [s]

# simulation
DT = 0.001  # simulation timestep [s]
T = 100  # total time [s]
TIME = np.linspace(start=0, stop=T, num=T * 1000)  # simulation time [s]

LEN_TIME = int(len(TIME))  # length of the simulation [ms]

# constants
EXP_M = np.exp(-DT / TAU_M)  # membrane exponential decay
EXP_IL = np.exp(-DT / TAU_IL)  # inhibitory learning exponential decay
EXP_I = np.exp(-DT / TAU_I)  # inhibitory synapses exponential decay
EXP_E = np.exp(-DT / TAU_E)  # excitatory synapses exponential decay


# DIRECTORY SETUP
# ============================================================================================================
path = " "

# UTILS (FUNCTIONS)
# ============================================================================================================


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
# 0 and 1 excitatory, 2 and 3 inhibitory
network = np.zeros((N, N), dtype="int32")
network[3, 1] = 1
network[1, 2] = 1
network[2, 0] = 1
network[0, 3] = 1


network = network.T  # to post pre

# VARIABLES SETUP
# ============================================================================================================

# SYNAPSES

w = np.zeros(shape=(N, N))
w[3, 1] = 1.0 * 1 / 10
w[1, 2] = 1.0 * 7
w[2, 0] = 1.0 * 1 / 10
w[0, 3] = 1.0 * 7

w = w.T  # from pre-post to post-pre

v = np.repeat(V_REST, N)  #  initial voltage


current = np.zeros(shape=(N))
current[0] = 2.1 / 10000
current[1] = 2.0 / 10000
current[2] = 2 / 100000
current[3] = 2 / 100000


# SIMULATION
# ============================================================================================================


@jit(nopython=True)
def run(v, w):
    # INITIAL
    t_ref = np.zeros(shape=N)
    i_i = np.zeros(shape=2)
    w_i = w
    s = np.zeros(shape=(N), dtype=np.int8)
    syn_c = np.zeros(shape=(N, N))
    factor = np.zeros(shape=(2, 2))
    syn_input = np.zeros(shape=(N))
    # outputs
    w = []
    spikes = []
    i_inputs = []
    for i in range(0, LEN_TIME, 1):
        # if i % (LEN_TIME / 10) == 0 and i != 0:
        #    print(i / LEN_TIME)
        t = TIME[i]
        # RECURRENT LAYER
        syn_input = np.sum(syn_c[:, :2], axis=1) * (v + V_E) - np.sum(
            syn_c[:, 2:], axis=1
        ) * (v + V_I)
        i_i *= EXP_IL
        i_i += np.sum(syn_c[2:, :2], axis=1)
        # voltage and co update
        v[t_ref >= t] = V_REST  # refractory period
        dvdt = (V_REST - v) + R * (-syn_input + current)
        dvdt *= DT / TAU_M
        v[t_ref < t] += dvdt[t_ref < t]
        s[v >= THETA_M] = 1
        t_ref[v >= THETA_M] = t + T_REF
        # update synaptic conductances
        syn_c[:, :2] *= EXP_E
        syn_c[:, 2:] *= EXP_I
        syn_c += GS * w_i * s
        # INHIBITORY LEARNING
        dw = ETA_I * (i_i - THETA_I) * s[2:]
        factor[:, dw > 0] = W_MAX - w_i[:2, 2:][:, dw > 0]
        factor[:, dw < 0] = w_i[:2, 2:][:, dw < 0]
        w_i[:2, 2:] += A_I * factor * dw
        w_i *= network  #  prevent new connections from forming
        # RECORDINGS - copies otherwise rewriting
        w.append(w_i[:2, 2:].copy())
        i_inputs.append(i_i.copy())
        spikes.append(s.copy())
        # reset for next iteration
        s[:] = 0
        factor[:] = 0

    return (
        w,
        spikes,
        i_inputs,
    )


tstart = datetime.datetime.now()

w, spikes, i_inputs = run(v, w)

spikes = np.array(spikes)
w = np.array(w)
i_inputs = np.array(i_inputs)


fr = get_network_firing_rates(spikes.T, 1000, LEN_TIME)

# save

np.save("fr_np.npy", fr)
np.save("w_np.npy", w[::1000])
np.save("i_i_np.npy", i_inputs)
