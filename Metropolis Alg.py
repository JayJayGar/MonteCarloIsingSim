import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import os
import time

# Create output folder for saved spin configurations
folder_name = "TestSpins"
os.makedirs(folder_name, exist_ok=True)

# L by L grid = N total spins
L = 40

REC_EVERY = L ** 2  # Record one per sweep instead of every run


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
def metropolis(spin_arr, times, BJ, energy, rec_every):
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
    n_records = times // rec_every
    net_spins = np.zeros(n_records)
    net_energy = np.zeros(n_records)
    L = spin_arr.shape[0]
    M = float(spin_arr.sum())
    rec = 0
    for t in range(times):
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

        if t % rec_every == 0:
            net_spins[rec] = M
            net_energy[rec] = energy
            rec += 1

    tests = spin_arr.flatten()

    return net_spins, net_energy, tests, energy

@numba.njit(cache=False)
def triangular_metropolis(spin_arr, times, BJ, energy, rec_every):
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
    n_records = times // rec_every
    net_spins = np.zeros(n_records)
    net_energy = np.zeros(n_records)
    L = spin_arr.shape[0]
    M = float(spin_arr.sum())
    rec = 0
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

        if t % rec_every == 0:
            net_spins[rec] = M
            net_energy[rec] = energy
            rec += 1

    tests = spin_arr.flatten()

    return net_spins, net_energy, tests, energy

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

def block_average(data, block_size=1500):
    """
    Compute block-averaged mean and variance to account for autocorrelations
    in Monte Carlo data. Grouping into blocks decorrelates the samples,
    giving a more accurate estimate of thermodynamic fluctuations.

    Parameters:
        data (ndarray): Time series of a thermodynamic observable.
        block_size (int): Number of steps per block.

    Returns:
        mean (float): Block-averaged mean.
        var (float): Block-averaged variance (dof=1).
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
    L = lattice.shape[0]
    Tc = LATTICE_PARAMS[lattice_shape]['Tc']
    energy = get_energy(lattice, lattice_shape)
    for i, bj in enumerate(BJs):
        T=1/bj
        # Steps scaled as L^2 to maintain constant sweep count across lattice sizes
        L_ref = 40
        N_ref_critical = 15_000_000
        N_ref_normal = 5_000_000

        if L >= 40:
            # L^(2+z) scaling - theoretically correct for large L
            n_steps_critical = int(N_ref_critical * (L / L_ref) ** 4.17)
            n_steps_normal = int(N_ref_normal * (L / L_ref) ** 4.17)
        else:
            # L^2 for smaller L's, shouldn't be noticeable
            n_steps_critical = int(N_ref_critical * (L / L_ref) ** 2)
            n_steps_normal = int(N_ref_normal * (L / L_ref) ** 2)

        n_steps = n_steps_critical if Tc * 0.8 <= T <= Tc * 1.2 else n_steps_normal

        spins, energies, tests, energy = (
            ALGORITHMS[lattice_shape](lattice, n_steps, bj, energy, REC_EVERY))

        # n_records is the size of the arrays based on how often we record data
        n_records = n_steps // REC_EVERY

        # Use data after x steps (post-equilibration)
        x = int(n_records * 0.10) if Tc * 0.8 <= T <= Tc * 1.2 else int(n_records * 0.50)
        eq_S = spins[x:].astype(np.float64)
        eq_E = energies[x:].astype(np.float64)

        #Larger blocks near Tc where autocorrelation time is long
        # block_size = 30 if Tc * 0.8 <= T <= Tc * 1.2 else 1
        block_size = 1
        # Block observables per spin
        M_mean, M_var = block_average(eq_S, block_size)             # <M_total>, Var(M_total)
        M_abs_mean, _ = block_average(np.abs(eq_S), block_size)     # <|M_total|>
        E_mean, E_var = block_average(eq_E, block_size)             # <E_total>, Var(E_total)
        M2_mean, _ = block_average(eq_S**2, block_size)             # <M^2_total>

        #Normalizations
        ms[i] = M_mean / L ** 2                         # <M> per spin
        ms_abs[i] = M_abs_mean / L ** 2                 # <|M|> per spin
        E_means[i] = E_mean / L ** 2                    # <E> per spin
        E_vars[i] = E_var / L ** 4                      # Var(E) per spin
        E_stds[i] = np.sqrt(E_vars[i])                  # σ(E) per spin


        # Save final spin configuration to file
        np.save(f"{folder_name}/spins{i}_run{run_index}", tests)

        # Thermodynamic observables
        C[i]= np.var(eq_E) / (T**2 * L**2)                      # C = Var(E_total) / (k_B T² N)
        chi[i] = np.var(eq_S) / (T * L**2)                      # χ = Var(M_total) / (k_B T N)
        chi_prime[i] = (M2_mean - M_abs_mean**2) / (T * L**2)   # χ' = (<M²> - <|M|>²) / (T N)

    return ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime


def get_spin_train_data(lattice, lattice_shape, BJs):
    """
    Generate labeled training data for a binary phase classifier.

    Labels:
        0 = Low temperature (ordered/ferromagnetic phase), T < Tc
        1 = High temperature (disordered/paramagnetic phase), T > Tc

    Parameters:
        lattice (ndarray): Initial 2D spin configuration.
        lattice_shape (str): Lattice geometry ('square' or 'triangular').
        BJs (array): Array of inverse temperatures.
    """
    Tc = LATTICE_PARAMS[lattice_shape]['Tc']
    lattice = lattice.copy()
    energy = get_energy(lattice, lattice_shape)
    for i, bj in enumerate(BJs):
        T = 1 / bj #type: ignore

        spins, energies, data, energy = (
            ALGORITHMS[lattice_shape](lattice, 1000000, bj, energy, REC_EVERY)) #type: ignore

        label = 0 if T < Tc else 1  # 0: ordered, 1: disordered

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
    Tc = LATTICE_PARAMS[lattice_shape]['Tc']
    lattice = lattice.copy()
    L_local = lattice.shape[0]
    n_steps_critical = int(10_000_000 * (L_local ** 2) // (40 ** 2))
    n_steps_normal = int(1_000_000 * (L_local ** 2) // (40 ** 2))
    energy = get_energy(lattice, lattice_shape)
    for i, bj in enumerate(BJs):
        T = 1 / bj #type: ignore

        # Use more steps near Tc where equilibration is slowest (critical slowing down)
        n_steps = n_steps_critical if Tc * 0.8 <= T <= Tc * 1.2 else n_steps_normal
        spins, energies, data, energy = (
            ALGORITHMS[lattice_shape](lattice, n_steps, bj, energy, REC_EVERY)) #type: ignore

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
    Tc = params['Tc']
    L = lattice.shape[0]
    Tc_prime = Tc + (1 / L)

    T = np.unique(np.concatenate([
        np.linspace(params['T_min'], Tc * 0.8, 15),
        np.linspace(Tc * 0.8, Tc * 1.2, 50),
        np.linspace(Tc * 1.2, params['T_max'], 15)
    ]))
    BJs = 1/T

    lattice = lattice.copy()
    ms, E_means, E_stds, E_vars, C, ms_abs, chi, chi_prime = get_spin_energy(lattice, BJs, run_index, lattice_shape)
    plot_misc(ms, E_means, E_stds, E_vars, C, ms_abs, T, Tc_prime, lattice_shape)
    plot_chis(chi, chi_prime, T, lattice_shape)

def plot_misc(ms, E_means, E_stds, E_vars, C, ms_abs, T, Tc_prime, lattice_shape):
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
        ax.axvline(Tc, color='r', linestyle='--', label=f"Tc = {Tc}")
        ax.axvline(Tc_prime, color='y', linestyle='-', label=f"$T_c'$ = {Tc_prime:.3f}")
        ax.legend(fontsize=8, loc='upper right')
        ax.text(0.20, 0.50, f'L = {L}', transform=ax.transAxes,
                fontsize=15, va='top', ha='left', color='black')

    axes[0, 0].plot(T, gaussian_filter1d(ms_abs, sigma=2))
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=True)
    axes[0].plot(T, chi)
    axes[0].set_ylabel('Chi')
    axes[0].set_xlabel('Temperature')
    axes[0].axvline(Tc, color='r', linestyle='--', label=f'$T_c$ = {Tc}')
    axes[0].legend(fontsize=12)

    axes[1].plot(T, chi_prime)
    axes[1].set_ylabel('Chi Prime')
    axes[1].set_xlabel('Temperature')
    axes[1].axvline(Tc, color='r', linestyle='--', label=f'$T_c$ = {Tc}')
    axes[1].legend(fontsize=12)

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
    runs = 30
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        # get_spin_train_data(lattice_ones, 'square', BJs)
        # get_spin_train_data(-lattice_ones, 'square', BJs)
        get_spin_test_data(lattice_ones, 'square', BJs)
        get_spin_test_data(-lattice_ones, 'square', BJs)

# Plot metropolis with positively charged lattice
lattice_plot(lattice_ones, 0, 'square')
# lattice_plot(lattice_ones, 0, 'triangular')

# Plot metropolis with negatively charged lattice
# lattice_plot(-lattice_ones, 0, 'triangular')

# toy_data('square')
# train_array = np.array(train_configs); np.save(f"{folder_name}/train_configs.npy", train_array)
# labels_array = np.array(train_labels); np.save(f"{folder_name}/train_labels.npy", labels_array)

# test_array = np.array(test_configs); np.save(f"{folder_name}/test_configs.npy", test_array)
# temp_array = np.array(test_temps); np.save(f"{folder_name}/test_temps.npy", temp_array)