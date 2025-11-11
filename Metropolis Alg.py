import numpy as np
import matplotlib.pyplot as plt
import numba
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
    return E * J / 2 #double counting

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
        # spin_i = spin_arr[x,y]

        xL = (x - 1) % N
        xR = (x + 1) % N
        yU = (y - 1) % N
        yD = (y + 1) % N
        dE = 2 * spin_arr[x, y] * (
                spin_arr[xL, y] + spin_arr[xR, y] +
                spin_arr[x, yU] + spin_arr[x, yD]
        )

        if (dE <= 0) or (np.random.random() < np.exp(-BJ * dE)):
            spin_arr[x, y] *= -1  #flip selected spin
            energy += dE

        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy

    return net_spins, net_energy


def get_spin_energy(lattice, BJs):
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
        spins, energies = metropolis(lattice, 600000, bj, get_energy(lattice))

        # Post EQ spins and energies
        eq_S = spins[-100000:]
        eq_E = energies[-100000:]
        T=1/bj


        ms[i] = eq_S.mean() / N ** 2 # Normalized magnetization / spin
        ms_abs[i] = np.mean(np.abs(eq_S)) / N ** 2 # Normalized absolute magnetization
        E_means[i] = eq_E.mean() / N ** 2
        E_stds[i] = eq_E.std() / N ** 2

        # Variances
        ms_vars[i] = eq_S.var()
        E_vars[i] = eq_E.var()

        #Heat capacity per spin -- k_b = 1
        C[i]= E_vars[i] / (T**2 * N**2)
        #Susceptibility chi
        chi[i] = ms_vars[i] / (T * N**2)
        #Expected chi'
        chi_prime[i] = (np.mean(eq_S**2) - np.mean(np.abs(eq_S))**2) / (T * N**2)

    return ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime




# plt.show(block=True)
# plt.plot(BJs, ms_n, label='Metropolis')
# plt.plot(BJs, E_stds_n)
# plt.title('Avg spin per Temp')
# plt.xlabel('Temperature')
# plt.show(block=True)

def lattice_plot(lattice):
    T = np.arange(0.5, 5, 0.1)
    BJs = 1/T
    global Tc
    Tc = 2.269  # critical temperature

    lattice = lattice.copy()
    ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime = get_spin_energy(lattice, BJs)
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

lattice_plot(lattice_p)