import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import scipy.ndimage as sp


# 50 by 50 grid
N = 50

init_random = np.random.random((N,N))
lattice_n = np.zeros((N,N))
lattice_n[init_random>=0.75] = 1
lattice_n[init_random<0.75] = -1

# print(init_random)
print(lattice_n)

init_random = np.random.random((N,N))
lattice_p = np.zeros((N,N))
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1


def get_energy(lattice):
    E, J = 0, 1
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

        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy

    return net_spins, net_energy


# spins, energies = metropolis(lattice_n, 1000000, 0.5, get_energy(lattice_n))
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#
# # Average spin plot
# ax = axes[0]
# ax.plot(spins / N**2)
# ax.set_xlabel('Algorithm Time Steps')
# ax.set_ylabel(r'Average Spin $\bar{m}$')
# ax.grid()
#
# # Energy plot
# ax = axes[1]
# ax.plot(energies)
# ax.set_xlabel('Algorithm Time Steps')
# ax.set_ylabel(r'Energy $E/J$')
# ax.grid()
#
# fig.tight_layout()
# fig.suptitle(r'Evolution of Average Spin and Energy for $\beta J=$0.7', y=1.07, size=18)
# plt.show()


def get_spin_energy(lattice, BJs):
    ms = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    for i, bj in enumerate(BJs):
        spins, energies = metropolis(lattice, 600000, bj, get_energy(lattice))
        ms[i] = spins[-100000:].mean() / N ** 2
        E_means[i] = energies[-100000:].mean() / N ** 2
        E_stds[i] = energies[-100000:].std() / N ** 2
    return ms, E_means, E_stds




# plt.show(block=True)
# plt.plot(BJs, ms_n, label='Metropolis')
# plt.plot(BJs, E_stds_n)
# plt.title('Avg spin per Temp')
# plt.xlabel('Temperature')
# plt.show(block=True)

def lattice_plot(lattice):
    BJs = np.arange(0.3, 1, 0.025)
    T = 1 / BJs
    lattice = lattice.copy()
    ms, E_means, E_stds = get_spin_energy(lattice, BJs)
    plt.plot(T, ms, label='Metropolis'); plt.title('Avg spin per Temp'); plt.show(block=True)
    plt.plot(T, E_means, label='Energy'); plt.title('Mean Energy per Temp'); plt.show(block=True)
    plt.plot(T,E_stds); plt.title('Standard Deviation per Temp'); plt.show(block=True)

lattice_plot(lattice_p)