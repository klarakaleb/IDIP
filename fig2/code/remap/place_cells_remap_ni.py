#%%
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import datetime

tstart = datetime.datetime.now()

seed = sys.argv[1]
seed = float(seed)
seed = int(seed)
np.random.seed(seed)



print(seed)


# PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------
# NEURON NUMBERS
N1 = 10
N1E = 10
N2 = 120
N2E = 100
N2I = 20
N = N1 + N2
N2E_NGROUPS = 10

# NEURON PROPERTIES
TAU_M = 20  # [ms]
V_REST = -60.0  # [mV]
R = 100  # [M_Ohm]
THETA_M = -50  # [V]
T_REF = 2  # [ms]
V_E = 0  # [mV]
V_I = -80  # [mV]

# SYNAPSES
TAU_E = 5  # [ms]
TAU_I = 10  # [ms]
TAU_IL = 2 * 4 * TAU_M
ETA_I = 0.01
ETA_E = 0.001
ETA_HOMEO = ETA_E * 0.1
A_E = 1
A_I = 1
A_HOMEO = 1
GS = 0.001  # 0.185
W = 1.0  # basic weight unit
WMAX = 1.5
THETA_I = 0.2
THETA_HOMEO = 5.2

# SIMULATION
N_RUNS = 3  # 100
TIME = np.arange(0, 30000, 1)
DT = 1  # [ms]

# PLACE DEPENDENT INPUT
A = 1
SIGMA = 21.8
PLACE_FIELDS = (np.arange(2, 30, 3)) * 0.01 * 100000
PLACE_FIELDS = PLACE_FIELDS.astype("int")

# CONSTANTS
GROUP_SIZE = int(N2E / N2E_NGROUPS)
LEN_TIME = len(TIME)
INPUT_CURRENT_CONSTANT = 2 * ((SIGMA * 0.01 * LEN_TIME) ** 2)
C_FRACTION = 0.2
SD = 0.05  # 0.005
EXP_M = np.exp(-DT / TAU_M)  # membrane exponential decay
EXP_IL = np.exp(-DT / TAU_IL)  # inhibitory learning exponential decay
EXP_I = np.exp(-DT / TAU_I)  # inhibitory synapses exponential decay
EXP_E = np.exp(-DT / TAU_E)  # excitatory synapses exponential decay

filename = "_trial.npy"

dirname = str(seed) + str(N_RUNS) + '_ni'

os.mkdir(dirname)


# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def get_network_firing_rates(spikes, window, time):
    firing_rate = np.zeros(shape=(len(spikes), int(len(time))))
    for i in range(0, len(time)):
        if i + window < len(spikes[0]):
            firing_rate[:, i] = np.sum(spikes[:, i : i + window], axis=1) / (
                window / 1000
            )
    return firing_rate


def create_network(N, N1, N1E, N2, N2E):
    n_input_synapses_per_neuron = (
        N1  #  full connectivity - so this function is abit over the top
    )
    connectivity_mat = np.zeros((N, N), dtype="int32")
    # connect excitatory to inputs
    for i in range(0, N2E):
        excitatory_inputs = np.random.choice(
            range(0, N1E), size=n_input_synapses_per_neuron, replace=False
        )
        for x in excitatory_inputs:
            connectivity_mat[x, i + N1] = 1
    # connect to inhibitory
    for i in range(0, N2E):
        connectivity_mat[(N1 + N2E) : (N1 + N2), i + N1] = 1  # I -> E
        connectivity_mat[i + N1, (N1 + N2E) : (N1 + N2)] = 1  # E -> I
    return connectivity_mat


# INPUT CURRENT GENERATION
# ----------------------------------------------------------------------------------------------------------------------

CA3_current = np.zeros(shape=(N1, LEN_TIME))
fr_scale = np.zeros(shape=(N1, LEN_TIME))

for i in range(0, LEN_TIME, DT):
    d = PLACE_FIELDS - i
    for x in range(N1):
        if d[x] > LEN_TIME / 2:
            d[x] = LEN_TIME - d[x]
        if d[x] < -LEN_TIME / 2:
            d[x] = LEN_TIME + d[x]
    fr_scale[:, i] = A * np.exp(-((d ** 2) / INPUT_CURRENT_CONSTANT))


CA3_current = fr_scale * C_FRACTION

# constant current to CA1 layer
CA1E_current = 0.05
CA1I_current = 0.1


# SYNAPSES
# ----------------------------------------------------------------------------------------------------------------------
network = create_network(N, N1, N1E, N2, N2E)  # (N1, N1E, N2, N2E)

CA3_CA1_mask = network[:N1, N1 : N1 + N2E]
CA1_EI_mask = network[N1 + N2E : N1 + N2, N1 : N1 + N2E]
CA1_IE_mask = network[N1 : N1 + N2E, N1 + N2E : N1 + N2]

CA3_CA1_w = np.zeros((N2E, N1))
CA3_CA1_w[:] = W
CA1_I_w = np.zeros((N2E, (N2 - N2E)))
CA1_I_w[:] = W
CA1_E_w = np.zeros(((N2 - N2E), N2E))
CA1_E_w[:] = W

# np.random.seed(seed)
noise = np.random.normal(0, SD, GROUP_SIZE)

# CA3->CA1 weights tuning
for i in range(int(N2E / GROUP_SIZE)):
    group_scale = fr_scale[i, PLACE_FIELDS]
    np.random.shuffle(noise)
    CA3_CA1_w[(i * GROUP_SIZE) : ((i + 1) * GROUP_SIZE), :] *= group_scale
    CA3_CA1_w[(i * GROUP_SIZE) : ((i + 1) * GROUP_SIZE), :] += noise[:, None]

CA3_CA1_w[CA3_CA1_w < 0] = 0.0  #  dale's law
# CA3_CA1_w[:] *= 1

CA1_E_w *= 2
CA1_I_w *= 1 / 1000


# CA3 INPUT CURRENT TO SPIKES (pregenerated)
# ----------------------------------------------------------------------------------------------------------------------

input_spikes = []

v = np.repeat(V_REST, N1)
s = np.zeros(shape=(N1), dtype=np.int8)
t_ref = np.zeros(shape=N1)

for i in range(0, LEN_TIME * 2, 1):
    if i % LEN_TIME == 0 and i != 0:
        refractory_overflow = np.where(t_ref >= LEN_TIME)
        if len(refractory_overflow) != 0:
            t_ref[refractory_overflow] = t_ref[refractory_overflow] % LEN_TIME
        refractory_underflow = np.where(t_ref < LEN_TIME)
        if len(refractory_underflow) != 0:
            t_ref[refractory_underflow] = 0
        input_spikes_0 = input_spikes.copy()
        input_spikes = []
    i = i % LEN_TIME
    t = TIME[i]
    dvdt = V_REST - v + R * CA3_current[:, i]
    dvdt *= DT / TAU_M
    dvdt[t < t_ref] = 0
    v += dvdt
    v[t < t_ref] = V_REST
    s[v >= THETA_M] = 1
    t_ref[v >= THETA_M] = t + T_REF
    input_spikes.append(s.copy())
    s[:] = 0

input_spikes_0 = np.array(input_spikes_0)
input_spikes_n = np.array(input_spikes)

"""
input_fr = get_network_firing_rates(input_spikes.T,1000,TIME)

fig = plt.figure()
ax = plt.axes()
for i in range(10):
    ax.plot(range(LEN_TIME),CA3_current[i,:])

fig = plt.figure()
ax = plt.axes()
for i in range(10):
    ax.plot(range(LEN_TIME),input_fr[i,:])

"""
#%%
# SETUP
# ----------------------------------------------------------------------------------------------------------------------

# RECORDINGS
# CA3_CA1_syn_c = []
# CA1_I_syn_c = []
# CA1_E_syn_c = []


# INITIAL


# SIMULATION
# ----------------------------------------------------------------------------------------------------------------------


def run(CA1_I_w, CA3_CA1_w):
    t_ref = np.zeros(shape=N2)
    v = np.repeat(V_REST, N2)
    i_i = np.zeros(shape=(N2 - N2E))
    s_t = np.zeros(shape=(N1 + N2E))
    CA1_E_syn_c_i = np.zeros(shape=((N2 - N2E), N2E))
    CA1_I_syn_c_i = np.zeros(shape=(N2E, (N2 - N2E)))
    CA3_CA1_syn_c_i = np.zeros(shape=(N2E, N1))
    w_i_i = CA1_I_w
    w_in_i = CA3_CA1_w
    s = np.zeros(shape=N2, dtype=np.int8)
    # outputs
    w_in = []
    w_i = []
    spikes = []
    i_inputs = []
    n_runs = 0
    input_spikes = input_spikes_0
    active = []
    for i in range(0, LEN_TIME * N_RUNS, 1):
        if i % LEN_TIME == 0 and i != 0:
            # spikes = np.stack(spikes)
            fr = get_network_firing_rates(np.array(spikes).T, 1000, TIME)
            if n_runs == 0:
                active = set(np.where(fr[:100, :] != 0)[0])
                active = np.array(list(active))
            np.save(dirname + "/" + str(n_runs) + "spikes" + filename, np.array(spikes))
            np.save(dirname + "/" + str(n_runs) + "fr" + filename, fr[:])
            np.save(
                dirname + "/" + str(n_runs) + "w_i" + filename, np.array(w_i)
            )
            np.save(
                dirname + "/" + str(n_runs) + "w_in" + filename, np.array(w_in)
            )
            np.save(
                dirname + "/" + str(n_runs) + "i_i" + filename,
                np.array(i_inputs)[::100],
            )
            refractory_overflow = np.where(t_ref >= LEN_TIME)
            if len(refractory_overflow) != 0:
                t_ref[refractory_overflow] = t_ref[refractory_overflow] % LEN_TIME
            refractory_underflow = np.where(t_ref < LEN_TIME)
            if len(refractory_underflow) != 0:
                t_ref[refractory_underflow] = 0
            spikes = []
            i_inputs = []
            w_i = []
            w_in = []
            input_spikes = input_spikes_n
            n_runs += 1
        # INTEGRATION
        i = i % LEN_TIME
        t = TIME[i]
        # INPUT/CA3
        # CA3_CA1_syn_c_i = CA3_CA1_syn_c_i * EXP_E + GS * w_in_i * input_spikes[i, :]
        CA3_CA1_syn_c_i = CA3_CA1_syn_c_i * EXP_E
        CA1_E_syn_c_i = CA1_E_syn_c_i * EXP_E
        CA1_I_syn_c_i = CA1_I_syn_c_i * EXP_I
        # CA1
        CA1E_syn_input = np.sum(CA3_CA1_syn_c_i, axis=1) * (v[:N2E] + V_E) - np.sum(
            CA1_I_syn_c_i, axis=1
        ) * (v[:N2E] + V_I)
        CA1I_syn_input = np.sum(CA1_E_syn_c_i, axis=1) * (v[N2E:] + V_E)
        i_i = i_i * EXP_IL + np.sum(CA1_E_syn_c_i, axis=1)
        dvdt_e = (V_REST - v[:N2E]) + R * (-CA1E_syn_input + CA1E_current)
        dvdt_i = (V_REST - v[N2E:]) + R * (-CA1I_syn_input + CA1I_current)
        dvdt = np.concatenate((dvdt_e, dvdt_i))
        dvdt *= DT / TAU_M
        dvdt[t < t_ref] = 0  #  refractory period
        v += dvdt
        v[t < t_ref] = V_REST
        if n_runs == 1:
            v[active] = V_REST
        s[v >= THETA_M] = 1
        t_ref[v >= THETA_M] = t + T_REF
        s_t[N1:] = s_t[N1:] * EXP_M + s[:N2E]
        CA3_CA1_syn_c_i += GS * w_in_i * input_spikes[i, :]
        CA1_E_syn_c_i += GS * CA1_E_w * s[:N2E]
        CA1_I_syn_c_i += GS * w_i_i * s[N2E:]
        s_t[:N1] = s_t[:N1] * EXP_M + input_spikes[i, :]
        # LEARNING
        if n_runs != 1:
            # INHIBITORY LEARNING
            delta_w = i_i - THETA_I
            w_i_i = w_i_i + (ETA_I * A_I * delta_w) * s[N2E:]
            # EXCITATORY LEARNING
            homeo_term = A_HOMEO * ETA_HOMEO * (np.sum(w_in_i, axis=1) - THETA_HOMEO)
            delta_w_post = (
                (WMAX - w_in_i) * A_E * ETA_E * s_t[None, :N1] - homeo_term[:, None]
            ) * s[:N2E, None]
            w_in_i = w_in_i + delta_w_post
            delta_w_pre = (
                (WMAX - w_in_i) * A_E * ETA_E * s_t[N1:, None] - homeo_term[:, None]
            ) * input_spikes[i, None]
            w_in_i = w_in_i + delta_w_pre
            w_in_i[w_in_i < 0] = 0.0
            w_i_i[w_i_i < 0] = 0.0
        # RECORDINGS
        w_in.append(w_in_i.copy())
        w_i.append(w_i_i.copy())
        i_inputs.append(i_i.copy())
        spikes.append(s.copy())
        s[:] = 0

    return w_in, w_i, spikes, i_inputs


CA1_I_w = np.load(
    "../../fig1/" + str(seed) + "100" + str(target) + "/w_i_trial.npy"
)[-1]
CA3_CA1_w = np.load(
    "../../fig1/" + str(seed) + "100" + str(target) + "/w_in_trial.npy"
)[-1]

w_in, w_i, spikes, inputs = run(CA1_I_w, CA3_CA1_w)


spikes = np.array(spikes)
w_in = np.array(w_in)
w_i = np.array(w_i)
inputs = np.array(inputs)

fr = get_network_firing_rates(spikes.T, 1000, TIME)
np.save(dirname + "/" + "spikes" + filename, spikes)
np.save(dirname + "/" + "fr" + filename, fr[:])
np.save(dirname + "/" + "w_i" + filename, w_i[:])
np.save(dirname + "/" + "w_in" + filename, w_in[:])
np.save(dirname + "/" + "i_i" + filename, inputs[::100])

tnow = datetime.datetime.now()
totalsecs = (tnow - tstart).total_seconds()

hrs = int(totalsecs // 3600)
mins = int(totalsecs % 3600) // 60
secs = int(totalsecs % 60)

timestamp = tnow.strftime("%b %d %Y %I:%M:%S %p").replace(" 0", " ")
print("{} ({} hrs {} mins {} secs elapsed)".format(timestamp, hrs, mins, secs))
