import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.ndimage import convolve, generate_binary_structure
import scipy.ndimage as sp
import os

folder_name = "TestSpins"
os.makedirs(folder_name, exist_ok=True)

# L by L grid = N
L = 30
global Tc
Tc = 2.269  # critical temperature

init_random = np.random.random((L,L))
lattice_n = np.zeros((L,L))
lattice_n[init_random>=0.75] = 1
lattice_n[init_random<0.75] = -1

init_random = np.random.random((L,L))
lattice_p = np.zeros((L,L))
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1

init_random = np.random.random((L,L))
lattice_r = np.zeros((L,L))
lattice_r[init_random>=0.50] = 1
lattice_r[init_random<0.50] = -1

lattice_ones = np.ones((L,L))

configs = []
labels = []

def get_energy(lattice):
    E, J = 0, 1
    right = np.roll(lattice, -1, axis=1)
    down = np.roll(lattice, -1, axis=0)
    return -J * np.sum(lattice * (right + down))

print(get_energy(lattice_ones))

@numba.njit
def metropolis(spin_arr, times, BJ, energy):
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0, times-1):
        # pick a random point on array and flip spin
        x = np.random.randint(0,L)
        y = np.random.randint(0,L)
        # spin_i = spin_arr[x,y]

        xL = (x - 1) % L
        xR = (x + 1) % L
        yU = (y - 1) % L
        yD = (y + 1) % L
        dE = 2 * spin_arr[x, y] * (
                spin_arr[xL, y] + spin_arr[xR, y] +
                spin_arr[x, yU] + spin_arr[x, yD]
        )

        if (dE <= 0) or (np.random.random() < np.exp(-BJ * dE)):
            spin_arr[x, y] *= -1  #flip selected spin
            energy += dE

        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy

    tests = spin_arr.flatten()

    return net_spins, net_energy, tests

@numba.njit
def triangular_metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    for t in range(0, times-1):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        if not (y & 1): #even
            neighbors = [
                spin_arr[(x-1)%L][(y-1)%L],
                spin_arr[(x-1)%L][y],
                spin_arr[x][(y-1)%L],
                spin_arr[x][(y+1)%L],
                spin_arr[(x+1)%L][(y-1)%L],
                spin_arr[(x+1)%L][y]
            ]
        else: #odd
            neighbors = [
                spin_arr[(x-1)%L][y],
                spin_arr[(x-1)%L][(y+1)%L],
                spin_arr[x][(y-1)%L],
                spin_arr[x][(y+1)%L],
                spin_arr[(x+1)%L][y],
                spin_arr[(x+1)%L][(y+1)%L]
            ]
        dE = 2 * spin_arr[x, y] * sum(neighbors)
        if (dE <= 0) or (np.random.random() < np.exp(-BJ * dE)):
            spin_arr[x, y] *= -1  # flip selected spin
            energy += dE

        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy

    tests = spin_arr.flatten()

    return net_spins, net_energy, tests

ALGORITHMS = {
    'square': metropolis,
    'triangular': triangular_metropolis,
    # "square-ice"
}


def get_spin_energy(lattice, BJs, run_index, lattice_shape):
    ms = np.zeros(len(BJs))
    ms_abs = np.zeros(len(BJs))
    ms_vars = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    E_vars = np.zeros(len(BJs))
    C = np.zeros(len(BJs))
    chi = np.zeros(len(BJs))
    chi_prime = np.zeros(len(BJs))
    for i, bj in enumerate(BJs):
        spins, energies, tests = ALGORITHMS[lattice_shape](lattice, 1000000, bj, get_energy(lattice))

        # Post EQ spins and energies
        eq_S = spins[-100000:]
        eq_E = energies[-100000:]
        T=1/bj


        ms[i] = eq_S.mean() / L ** 2 # Normalized magnetization / spin
        ms_abs[i] = np.mean(np.abs(eq_S)) / L ** 2 # Normalized absolute magnetization
        E_means[i] = eq_E.mean() / L ** 2
        E_stds[i] = eq_E.std() / L ** 2

        np.save(f"{folder_name}/spins{i}_run{run_index}", tests)

        # Variances
        ms_vars[i] = eq_S.var()
        E_vars[i] = eq_E.var()

        #Heat capacity per spin -- k_b = 1
        C[i]= E_vars[i] / (T**2 * L**2)
        #Susceptibility chi
        chi[i] = ms_vars[i] / (T * L**2)
        #Expected chi'
        chi_prime[i] = (np.mean(eq_S**2) - np.mean(np.abs(eq_S))**2) / (T * L**2)

    return ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime


def get_spin_toy_data(lattice, run_index, lattice_shape):
    T = np.arange(0.5, 5, 0.1)
    BJs = 1 / T

    for i, bj in enumerate(BJs):
        T = 1 / bj

        if T >= Tc * 0.8 and T <= Tc * 1.2:
            continue

        spins, energies, tests = ALGORITHMS[lattice_shape](lattice, 100000, bj, get_energy(lattice))

        # Assign label
        if T < Tc * 0.8:
            label = 0  # Low temperature
        else:  # T > Tc * 1.2
            label = 1  # High temperature

        configs.append(tests)
        labels.append(label)

# plt.show(block=True)
# plt.plot(BJs, ms_n, label='Metropolis')
# plt.plot(BJs, E_stds_n)
# plt.title('Avg spin per Temp')
# plt.xlabel('Temperature')
# plt.show(block=True)

def lattice_plot(lattice, run_index, lattice_shape):
    T = np.arange(0.5, 5, 0.1)
    BJs = 1/T

    lattice = lattice.copy()
    ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime = get_spin_energy(lattice, BJs, run_index, lattice_shape)
    plot_misc(ms, E_means, E_stds, E_vars, C, ms_abs, T)
    plot_chis(chi, chi_prime, T)



def plot_misc(ms, E_means, E_stds, E_vars, C, ms_abs, T):
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), sharex=True)

    axes[0, 0].plot(T, ms)
    axes[0, 0].set_ylabel('Avg spin per site')
    axes[0, 0].set_title('Average Spin vs Temperature')
    axes[0, 0].axvline(Tc, color='r', linestyle='--')

    axes[1, 0].plot(T, E_means)
    axes[1, 0].set_ylabel('Mean Energy per site')
    axes[1, 0].set_title('Energy vs Temperature')
    axes[1, 0].axvline(Tc, color='r', linestyle='--')

    axes[2, 0].plot(T, E_stds)
    axes[2, 0].set_ylabel('Energy Std Dev')
    axes[2, 0].set_xlabel('Temperature')
    axes[2, 0].set_title('Energy Fluctuations vs Temperature')
    axes[2, 0].set_ylim(0, 0.8)
    axes[2, 0].axvline(Tc, color='r', linestyle='--')

    axes[0, 1].plot(T, C)
    axes[0, 1].set_ylabel('C')
    axes[0, 1].set_title('Heat Fluctuations vs Temperature')
    axes[0, 1].axvline(Tc, color='r', linestyle='--')

    axes[1, 1].plot(T, ms_abs)
    axes[1, 1].set_ylabel('<|M|>')
    axes[1, 1].set_title('Absolute Magnetization vs Temperature')
    axes[1, 1].axvline(Tc, color='r', linestyle='--')

    fig.tight_layout()
    plt.show()

def plot_chis(chi, chi_prime, T):
    fig, axes = plt.subplots(1, 2, figsize=(8, 10), sharex=True)
    axes[0].plot(T, chi)
    axes[0].set_ylabel('Chi')
    axes[0].set_xlabel('Temperature')
    axes[0].axvline(Tc, color='r', linestyle='--')

    axes[1].plot(T, chi_prime)
    axes[1].set_ylabel('Chi Prime')
    axes[1].set_xlabel('Temperature')
    axes[1].axvline(Tc, color='r', linestyle='--')

    fig.tight_layout()
    plt.show()

# for i in range(5):
#     lattice_plot(lattice_ones, i+1, 'square')
#     lattice_plot(-lattice_ones, -i, 'square')

def get_tests():
    for i in range(10):
        get_spin_toy_data(lattice_ones, 0, 'square')
        get_spin_toy_data(-lattice_ones, 0, 'square')



get_tests()
configs_array = np.array(configs); np.save(f"{folder_name}/train_configs.npy", configs_array)
labels_array = np.array(labels); np.save(f"{folder_name}/train_labels.npy", labels_array)