import matplotlib.pyplot as plt
import numpy as np

## FIGURE 5(A), Fully implented NN with best weights
epsilon = 1/3
x_vals = np.array([-1,1])
N=900
randspins = 10000

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
configs_down = np.random.choice([-1, 1], size=(randspins//3, N), p=[0.98, 0.02])
configs_neutral = np.random.choice([-1, 1], size=(randspins//3, N), p=[0.50, 0.50])
configs_up = np.random.choice([-1, 1], size=(randspins//3, N), p=[0.02, 0.98])

spin_configs = np.vstack([configs_down, configs_neutral, configs_up])

m = np.mean(spin_configs, axis=1)


# 0 mean, unit std
W0 = np.random.normal(0, 1, size=(3, N))
b0 = np.random.normal(0, 1, size=(3, 1))

activations = W0 @ spin_configs.T + b0

print(activations)

plt.figure(figsize=(10, 4))


plt.scatter(m, activations[0, :],
            label='Neuron 1', color='blue', alpha=0.5, s=10)
plt.scatter(m, activations[1, :],
            label='Neuron 2', color='red', alpha=0.5, s=10)
plt.scatter(m, activations[2, :],
            label='Neuron 3', color='green', alpha=0.5, s=10)

plt.show()