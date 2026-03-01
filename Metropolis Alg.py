import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.ndimage import convolve, generate_binary_structure
import scipy.ndimage as sp
import os
import time

# Create output folder for saved spin configurations
folder_name = "TestSpins"
os.makedirs(folder_name, exist_ok=True)

# L by L grid = N total spins
L = 30

# --- Lattice Initialization ---
# Negatively biased: 75% spins down (-1)
init_random = np.random.random((L,L))
lattice_n = np.zeros((L,L))
lattice_n[init_random>=0.75] = 1
lattice_n[init_random<0.75] = -1

# Positively biased: 75% spins up (+1)
init_random = np.random.random((L,L))
lattice_p = np.zeros((L,L))
lattice_p[init_random>=0.25] = 1
lattice_p[init_random<0.25] = -1

# Random: 50/50 spins up and down
init_random = np.random.random((L,L))
lattice_r = np.zeros((L,L))
lattice_r[init_random>=0.50] = 1
lattice_r[init_random<0.50] = -1

# Fully ordered lattice: all spins up (+1)
lattice_ones = np.ones((L,L))

# Storage for ML training and test data
train_configs = []
train_labels = []
test_configs = []
test_temps = []

def get_energy(lattice, lattice_shape):
    """
    Compute the total energy of the lattice using the Ising Hamiltonian H = -J * sum(S_i * S_j).
    Uses periodic boundary conditions via np.roll. Ran through once to avoid double counting.

    Parameters:
        lattice (ndarray): 2D array of spin values (+1 or -1).
        lattice_shape (str): Lattice geometry, either 'square' or 'triangular'.

    Returns:
        float: Total energy of the lattice.

    Raises:
        Exception: If an unsupported lattice shape is provided.
    """
    E, J = 0, 1
    right = np.roll(lattice, -1, axis=1)    # Neighbor to the right
    down = np.roll(lattice, -1, axis=0)     # Neighbor below

    if lattice_shape == "square":
        # Each spin interacts with its right and down neighbors (avoids double-counting)
        return -J * np.sum(lattice * (right + down))
    elif lattice_shape == "triangular":
        # Triangular lattice has two additional diagonal neighbors
        diag_down_right = np.roll(np.roll(lattice, -1, axis=0), -1, axis=1)
        diag_down_left = np.roll(np.roll(lattice, -1, axis=0), 1, axis=1)
        return -J * np.sum(lattice * (right + down + diag_down_right + diag_down_left))

    raise Exception("Shape wrong or not implemented")

# print(get_energy(lattice_ones, "square")
# print(get_energy(lattice_ones, "triangular")

@numba.njit(cache=False)
def metropolis(spin_arr, times, BJ, energy):
    """
    Run the Metropolis-Hastings algorithm on a square lattice.
    At each step, a random spin is selected and flipped if the move is energetically
    favourable or accepted probabilistically via the Boltzmann factor.
    JIT-compiled with Numba for performance.

    Parameters:
        spin_arr (ndarray): 2D array of spin values (+1 or -1).
        times (int): Number of Monte Carlo steps to perform.
        BJ (float): Dimensionless inverse temperature (beta * J = J / k_B T).
        energy (float): Initial energy of the lattice.

    Returns:
        net_spins (ndarray): Total magnetization at each step.
        net_energy (ndarray): Total energy at each step.
        tests (ndarray): Flattened final spin configuration.
    """
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    M = float(spin_arr.sum())
    for t in range(0, times-1):
        # Pick a random spin site
        x = np.random.randint(0,L)
        y = np.random.randint(0,L)

        # Periodic boundary conditions for neighbors
        xL = (x - 1) % L; xR = (x + 1) % L;
        yU = (y - 1) % L; yD = (y + 1) % L

        # Energy change if this spin were flipped
        dE = 2 * spin_arr[x, y] * (
                spin_arr[xL, y] + spin_arr[xR, y] +
                spin_arr[x, yU] + spin_arr[x, yD]
        )

        # Accept flip if energy decreases or by Boltzmann probability
        if (dE <= 0) or (np.random.random() < np.exp(-BJ * dE)):
            M -= 2 * spin_arr[x, y]
            spin_arr[x, y] *= -1  #flip selected spin
            energy += dE

        net_spins[t] = M
        net_energy[t] = energy

    tests = spin_arr.flatten()

    return net_spins, net_energy, tests

@numba.njit(cache=False)
def triangular_metropolis(spin_arr, times, BJ, energy):
    """
    Run the Metropolis-Hastings algorithm on a triangular lattice.
    Each spin has 6 neighbors; the neighbor pattern alternates based on
    whether the column index is even or odd.
    JIT-compiled with Numba for performance.

    Parameters:
        spin_arr (ndarray): 2D array of spin values (+1 or -1).
        times (int): Number of Monte Carlo steps to perform.
        BJ (float): Dimensionless inverse temperature (beta * J = J / k_B T).
        energy (float): Initial energy of the lattice.

    Returns:
        net_spins (ndarray): Total magnetization at each step.
        net_energy (ndarray): Total energy at each step.
        tests (ndarray): Flattened final spin configuration.
    """
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    M = float(spin_arr.sum())
    for t in range(0, times-1):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)

        # Neighbor layout depends on even/odd column (offset rows in triangular geometry)
        if not (y & 1): # Even column
            neighbors = [
                spin_arr[(x-1)%L][(y-1)%L],
                spin_arr[(x-1)%L][y],
                spin_arr[x][(y-1)%L],
                spin_arr[x][(y+1)%L],
                spin_arr[(x+1)%L][(y-1)%L],
                spin_arr[(x+1)%L][y]
            ]
        else: # Odd column
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
            M -= 2 * spin_arr[x, y]
            spin_arr[x, y] *= -1  # flip selected spin
            energy += dE

        net_spins[t] = M
        net_energy[t] = energy

    tests = spin_arr.flatten()

    return net_spins, net_energy, tests

# Maps lattice geometry names to their corresponding Metropolis algorithm
ALGORITHMS = {
    'square': metropolis,
    'triangular': triangular_metropolis,
    # "square-ice"
}

# Adjust Temperature min and maxes to adjust graphs
LATTICE_PARAMS = {
    'square': {
        'T_min': 1.0,
        'T_max': 3.5,
        'n_points': 50,
        'Tc': 2.269
    },
    'triangular': {
        'T_min': 2.0,
        'T_max': 5.5,
        'n_points': 75,
        'Tc': 3.641  # 4/ln(3)
    }
}

def block_average(data, block_size=1000):
    """
    Compute block-averaged mean and variance to account for autocorrelations
    in Monte Carlo data. Grouping into blocks decorrelates the samples,
    giving a more accurate estimate of thermodynamic fluctuations.

    Parameters:
        data (ndarray): Time series of a thermodynamic observable.
        block_size (int): Number of steps per block.

    Returns:
        mean (float): Block-averaged mean.
        var (float): Block-averaged variance (ddof=1).
    """
    n_blocks = len(data) // block_size
    blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size)
    block_means = np.mean(blocks, axis=1)
    return np.mean(block_means), np.var(block_means, ddof=1)


def get_spin_energy(lattice, BJs, run_index, lattice_shape):
    """
    Run the Metropolis algorithm across a range of temperatures and compute
    thermodynamic observables from the equilibrated portion of each simulation.

    Parameters:
        lattice (ndarray): Initial 2D spin configuration.
        BJs (array): Array of inverse temperatures (beta * J = J / k_B T).
        run_index (int): Index used for labeling saved spin files.
        lattice_shape (str): Lattice geometry ('square' or 'triangular').

    Returns:
        ms (ndarray): Mean magnetization per spin.
        E_means (ndarray): Mean energy per spin.
        E_stds (ndarray): Standard deviation of energy per spin.
        E_vars (ndarray): Variance of energy per spin².
        C (ndarray): Specific heat capacity per spin.
        ms_abs (ndarray): Mean absolute magnetization per spin.
        chi (ndarray): Magnetic susceptibility.
        chi_prime (ndarray): Alternative susceptibility using <|M|>.
    """
    ms = np.zeros(len(BJs))
    ms_abs = np.zeros(len(BJs))
    E_means = np.zeros(len(BJs))
    E_stds = np.zeros(len(BJs))
    E_vars = np.zeros(len(BJs))
    C = np.zeros(len(BJs))
    chi = np.zeros(len(BJs))
    chi_prime = np.zeros(len(BJs))
    Tc = LATTICE_PARAMS[lattice_shape]['Tc']
    for i, bj in enumerate(BJs):
        T=1/bj
        n_steps = 3000000 if Tc * 0.8 <= T <= Tc * 1.2 else 1500000

        spins, energies, tests = (
            ALGORITHMS[lattice_shape](lattice, n_steps, bj, get_energy(lattice, lattice_shape)))

        # Use data after x steps (post-equilibration)
        x = int(n_steps * 0.15)
        eq_S = spins[x:]
        eq_E = energies[x:]

        # Block observables per spin
        M_mean, M_var = block_average(eq_S)             # <M_total>, Var(M_total)
        M_abs_mean, _ = block_average(np.abs(eq_S))     # <|M_total|>
        E_mean, E_var = block_average(eq_E)             # <E_total>, Var(E_total)
        M2_mean, _ = block_average(eq_S**2)             # <M^2_total>

        #Normalizations
        ms[i] = M_mean / L ** 2                         # <M> per spin
        ms_abs[i] = M_abs_mean / L ** 2                 # <|M|> per spin
        E_means[i] = E_mean / L ** 2                    # <E> per spin
        E_vars[i] = E_var / L ** 4                      # Var(E) per spin
        E_stds[i] = np.sqrt(E_vars[i])                  # σ(E) per spin


        # Save final spin configuration to file
        np.save(f"{folder_name}/spins{i}_run{run_index}", tests)

        # Thermodynamic observables
        C[i]= E_var / (T**2 * L**2)                     # C = Var(E_total) / (k_B T² N)
        chi[i] = M_var / (T * L**2)                     # χ = Var(M_total) / (k_B T N)
        chi_prime[i] = (M2_mean - M_abs_mean**2) / (T * L**2)   # χ' = (<M²> - <|M|>²) / (T N)

    return ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime


def get_spin_train_data(lattice, lattice_shape, BJs):
    """
    Generate labeled training data for a binary phase classifier.
    Skips temperatures within 20% of Tc (the critical region) to avoid
    ambiguous near-transition samples.

    Labels:
        0 = Low temperature (ordered/ferromagnetic phase), T < 0.8 * Tc
        1 = High temperature (disordered/paramagnetic phase), T > 1.2 * Tc

    Parameters:
        lattice (ndarray): Initial 2D spin configuration.
        lattice_shape (str): Lattice geometry ('square' or 'triangular').
        BJs (array): Array of inverse temperatures.
    """
    Tc = LATTICE_PARAMS[lattice_shape]['Tc']
    for i, bj in enumerate(BJs):
        T = 1 / bj #type: ignore

        # Skip the critical region to keep training labels clean
        if Tc * 0.85 <= T <= Tc * 1.15:
            continue

        lattice = lattice.copy()
        spins, energies, data = (
            ALGORITHMS[lattice_shape](lattice, 1000000, bj, get_energy(lattice, lattice_shape))) #type: ignore

        label = 0 if T < Tc * 0.85 else 1  # 0: ordered, 1: disordered

        train_configs.append(data)
        train_labels.append(label)

def get_spin_test_data(lattice, lattice_shape, BJs):
    """
    Generate unlabelled test data across the full temperature range.
    Runs longer simulations near the critical region (1.8 <= T <= 2.8)
    to improve equilibration where fluctuations are largest.

    Parameters:
        lattice (ndarray): Initial 2D spin configuration.
        lattice_shape (str): Lattice geometry ('square' or 'triangular').
        BJs (array): Array of inverse temperatures.
    """
    for i, bj in enumerate(BJs):
        T = 1 / bj #type: ignore

        # Use more steps near Tc where equilibration is slowest (critical slowing down)
        Tc = LATTICE_PARAMS[lattice_shape]['Tc']
        n_steps = 5000000 if Tc * 0.8 <= T <= Tc * 1.2 else 1000000

        lattice = lattice.copy()
        spins, energies, data = (
            ALGORITHMS[lattice_shape](lattice, n_steps, bj, get_energy(lattice, lattice_shape))) #type: ignore

        test_configs.append(data)
        test_temps.append(T)

def lattice_plot(lattice, run_index, lattice_shape):
    """
    Run a full simulation sweep over temperatures and produce all diagnostic plots.

    Parameters:
        lattice (ndarray): Initial 2D spin configuration.
        run_index (int): Index used for labeling saved spin files.
        lattice_shape (str): Lattice geometry ('square' or 'triangular').
    """
    params = LATTICE_PARAMS[lattice_shape]
    T = np.linspace(params['T_min'], params['T_max'], params['n_points'])
    BJs = 1/T

    lattice = lattice.copy()
    ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime = get_spin_energy(lattice, BJs, run_index, lattice_shape)
    plot_misc(ms, E_means, E_stds, E_vars, C, ms_abs, T, lattice_shape)
    plot_chis(chi, chi_prime, T, lattice_shape)



def plot_misc(ms, E_means, E_stds, E_vars, C, ms_abs, T, lattice_shape):
    """
    Plot magnetization, energy, energy fluctuations, specific heat,
    and absolute magnetization as functions of temperature.
    A vertical dashed red line marks the critical temperature Tc.

    Parameters:
        ms (ndarray): Mean magnetization per spin.
        E_means (ndarray): Mean energy per spin.
        E_stds (ndarray): Energy standard deviation per spin.
        E_vars (ndarray): Energy variance.
        C (ndarray): Specific heat.
        ms_abs (ndarray): Absolute magnetization per spin.
        T (ndarray): Temperature values.
        lattice_shape (str): Lattice geometry ('square' or 'triangular')
    """
    Tc = LATTICE_PARAMS[lattice_shape]['Tc']
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), sharex=True)
    fig.suptitle(f'{lattice_shape.capitalize()} Data')
    for ax in axes.flat:
        ax.tick_params(labelbottom=True)
        ax.axvline(Tc, color='r', linestyle='--')
        ax.text(Tc, -0.1, f'$T_c$', ha='center', fontsize=12,
                fontweight='bold', color='r', transform=ax.get_xaxis_transform())

    axes[0, 0].plot(T, ms_abs)
    axes[0, 0].set_ylabel('<|M|>')
    axes[0, 0].set_title('Absolute Magnetization vs Temperature')

    axes[1, 0].plot(T, E_means)
    axes[1, 0].set_ylabel('Mean Energy per site')
    axes[1, 0].set_title('Energy vs Temperature')

    axes[2, 0].plot(T, E_stds)
    axes[2, 0].set_ylabel('Energy Std Dev')
    axes[2, 0].set_xlabel('Temperature')
    axes[2, 0].set_title('Energy Fluctuations vs Temperature')
    axes[2, 0].set_ylim(0, 0.8)

    axes[0, 1].plot(T, C)
    axes[0, 1].set_ylabel('C')
    axes[0, 1].set_title('Heat Fluctuations vs Temperature')

    axes[1, 1].plot(T, ms)
    axes[1, 1].set_ylabel('Avg spin per site')
    axes[1, 1].set_title('Polarized Spin vs Temperature')

    fig.tight_layout()
    plt.show()

def plot_chis(chi, chi_prime, T, lattice_shape):
    """
    Plot magnetic susceptibility (chi) and the alternative susceptibility (chi)
    as functions of temperature, with a vertical line at Tc.

    Parameters:
        chi (ndarray): Susceptibility from variance of M.
        chi_prime (ndarray): Susceptibility from connected correlator.
        T (ndarray): Temperature values.
        lattice_shape (str): Lattice geometry ('square' or 'triangular')
    """
    Tc = LATTICE_PARAMS[lattice_shape]['Tc']
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

def toy_data(lattice_shape):
    """
    Helper function to generate and save test and training data
    over a temperature range for use in ML experiments.
    Training data generation is currently commented out.

    Parameters:
        lattice_shape (str): Lattice geometry ('square' or 'triangular').
    """
    params = LATTICE_PARAMS[lattice_shape]
    T_range = np.linspace(params['T_min'], params['T_max'], params['n_points'])
    BJs = 1 / T_range
    for i in range(30):
        print(f"Run {i+1}/10")
        # get_spin_train_data(lattice_ones, 'square', BJs)
        # get_spin_train_data(-lattice_ones, 'square', BJs)
        # get_spin_test_data(lattice_ones, 'square', BJs)
        # get_spin_test_data(-lattice_ones, 'square', BJs)

# Plot metropolis with positively charged lattice
lattice_plot(lattice_ones, 0, 'square')
# lattice_plot(lattice_ones, 0, 'triangular')

# Plot metropolis with negatively charged lattice
# lattice_plot(-lattice_ones, 0, 'triangular')

# toy_data('triangular')
# train_array = np.array(train_configs); np.save(f"{folder_name}/train_configs.npy", train_array)
# labels_array = np.array(train_labels); np.save(f"{folder_name}/train_labels.npy", labels_array)

# test_array = np.array(test_configs); np.save(f"{folder_name}/test_configs.npy", test_array)
# temp_array = np.array(test_temps); np.save(f"{folder_name}/test_temps.npy", temp_array)