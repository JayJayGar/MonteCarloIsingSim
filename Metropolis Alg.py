import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import scipy.ndimage as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 50 by 50 grid
N = 50
plt.ion()

init_random = np.random.random((N,N))
lattice_n = np.zeros((N,N))
lattice_n[init_random>=0.75] = 1
lattice_n[init_random<0.75] = -1

print(init_random)
print(lattice_n)

init_random = np.random.random((N,N))
lattice_p = np.zeros((N,N))
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1

# print(init_random)
# print(lattice_p)
# plt.imshow(lattice_p)
# plt.show(block=False)

def get_energy(lattice):
    E, J = 0
    # interaction with both left and right neighbors
    E -= np.sum(lattice[:, :-1] * lattice[:, 1:])
    # interaction with both top and bottom neighbors
    E -= np.sum(lattice[:-1, :] * lattice[1:, :])
    return E * J

print(get_energy(lattice_p))

@numba.njit
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0, times-1):
        # pick a random point on array and flip spin
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_arr[x,y] #init spin
        spin_f = spin_i*-1 #proposed spin flip

        # compute change in energy
        E_i = 0
        E_f = 0
        if x > 0:
            E_i += -spin_i * spin_arr[x - 1, y]
            E_f += -spin_f * spin_arr[x - 1, y]
        if x < N - 1:
            E_i += -spin_i * spin_arr[x + 1, y]
            E_f += -spin_f * spin_arr[x + 1, y]
        if y > 0:
            E_i += -spin_i * spin_arr[x, y - 1]
            E_f += -spin_f * spin_arr[x, y - 1]
        if y < N - 1:
            E_i += -spin_i * spin_arr[x, y + 1]
            E_f += -spin_f * spin_arr[x, y + 1]

        # 3 / 4. change state with designated probabilities
        dE = E_f - E_i
        if (dE <= 0) or (np.random.random() < np.exp(-BJ * dE)): #CHECK, changed to <, and or instead of *
            spin_arr[x, y] = spin_f
            energy += dE
        # elif dE <= 0: #CHECK
        #     spin_arr[x, y] = spin_i
        #     energy += dE

        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy

    return net_spins, net_energy


spins, energies = metropolis(lattice_n, 1000000, 0.2, get_energy(lattice_n))

fig = make_subplots(rows=1, cols=2)

# Average spin plot
fig.add_trace(go.Scatter(y=spins / N ** 2, mode='lines', line=dict(color='blue')), row=1, col=1)

# Energy plot
fig.add_trace(go.Scatter(y=energies, mode='lines', line=dict(color='red')), row=1, col=2)

fig.update_xaxes(title_text="Time Steps", row=1, col=1)
fig.update_xaxes(title_text="Time Steps", row=1, col=2)
fig.update_yaxes(title_text="Average Spin", row=1, col=1)
fig.update_yaxes(title_text="Energy", row=1, col=2)
fig.update_layout(title=f'βJ = {BJ}', height=400, showlegend=False)
fig.show()


def get_spin_energy(lattice, BJs):
    ms = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    for i, bj in enumerate(BJs):
        spins, energies = metropolis(lattice, 1000000, bj, get_energy(lattice))
        ms[i] = spins[-100000:].mean() / N ** 2
        E_means[i] = energies[-100000:].mean()
        E_stds[i] = energies[-100000:].std()
    return ms, E_means, E_stds


BJs = np.arange(0.1, 2, 0.05)
ms_n, E_means_n, E_stds_n = get_spin_energy(lattice_n, BJs)
ms_p, E_means_p, E_stds_p = get_spin_energy(lattice_p, BJs)