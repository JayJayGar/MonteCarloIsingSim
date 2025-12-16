import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.ndimage import zoom

## FIGURE 5(A), Fully implented NN with best weights
epsilon = 1/3
x_vals = np.array([-1,1])
randspins = 1000

def neuron1(m):
    return (m - epsilon) / (1 + epsilon)

def neuron2(m):
    return (-m - epsilon) / (1 + epsilon)

def neuron3(m):
    return (m + epsilon) / (1 + epsilon)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, neuron1(x_vals), label='Neuron 1: (m-ε)/(1+ε)', color='black', linewidth=2)
plt.plot(x_vals, neuron2(x_vals), label='Neuron 2: (-m-ε)/(1+ε)', color='green', linewidth=2)
plt.plot(x_vals, neuron3(x_vals), label='Neuron 3: (m+ε)/(1+ε)', color='blue', linewidth=2)

plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

plt.xlabel('m(x)', fontsize=12)
plt.ylabel('Wx + b', fontsize=12)
plt.title(f'Hidden Layer Arguments vs Magnetization (ε=1/3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.show()

## FIGURE 5(B) random initial weights, should represent a Bell curve scatterplot

## Mini-batches with col size N - spin_configs is random ising models
# block = randspins // 3
# block_per_group = block // 3
#
# spin_configs = np.zeros((block, N))
#
# # Group 1: magnetization around -1 (mostly down spins)
# for i in range(block_per_group):
#     p_down = np.random.uniform(0.90, 1.0)  # 90-100% down spins
#     spin_configs[i, :] = np.random.choice([-1, 1], size=N, p=[p_down, 1-p_down])
#
# # Group 2: magnetization around 0 (balanced)
# for i in range(block_per_group, 2 * block_per_group):
#     p_down = np.random.uniform(0.45, 0.55)  # 45-55% down spins (near 50/50)
#     spin_configs[i, :] = np.random.choice([-1, 1], size=N, p=[p_down, 1-p_down])
#
# # Group 3: magnetization around +1 (mostly up spins)
# for i in range(2 * block_per_group, block):
#     p_down = np.random.uniform(0.0, 0.10)  # 0-10% down spins (90-100% up)
#     spin_configs[i, :] = np.random.choice([-1, 1], size=N, p=[p_down, 1-p_down])
#
# m = np.mean(spin_configs, axis=1)


configs_list = []
file_list = sorted(glob.glob("TestSpins/spins*.npy"))

for file_path in file_list:
    try:
        data = np.load(file_path)
        configs_list.append(data.flatten())
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

spin_configs = np.array(configs_list)
print(spin_configs.shape)
N = spin_configs.shape[1]

print(spin_configs)

m = np.mean(spin_configs, axis=1)
sorted_idx = np.argsort(m)
m_sorted = m[sorted_idx]


# 0 mean, unit std
W0 = np.random.normal(0, 1, size=(3, N))
b0 = np.random.normal(0, 1, size=(3, 1))

activations = W0 @ spin_configs.T + b0

print(activations)

plt.figure(figsize=(10, 4))


plt.scatter(m_sorted, activations[0, :],
            label='Neuron 1', color='blue', alpha=0.5, s=10)
plt.scatter(m_sorted, activations[1, :],
            label='Neuron 2', color='red', alpha=0.5, s=10)
plt.scatter(m_sorted, activations[2, :],
            label='Neuron 3', color='green', alpha=0.5, s=10)

plt.show()