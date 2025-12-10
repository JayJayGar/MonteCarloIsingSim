import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, simplify, latex, pprint

m_x = 0.1
epsilon =0.1
N = 3

x = torch.tensor([+1.0,+1.0,+1.0]) #all spins in this scenario, all spins
m = float(torch.mean(x)) # sigma(x) representation equals m(x)

W = torch.zeros(3, N)
W[0, :] = 1.0
W[1, :] = -1.0
W[2, :] = 1.0
W = W / (N * (1 + epsilon))
print(W)

b = torch.tensor([-1,-1,1])
b = b * epsilon / (1 + epsilon)
print(b)

# TIP matrix multiplication is @ or
# Wx_b = torch.matmul(W, x) + b

Wx_b = torch.tensor([
    (m - epsilon) / (1 + epsilon),      # Neuron 1
    (-m - epsilon) / (1 + epsilon),     # Neuron 2
    (m + epsilon) / (1 + epsilon)       # Neuron 3
]).unsqueeze(1)

print(Wx_b)

Wx_b = (W @ x + b).unsqueeze(1)
print(Wx_b)

m, E = symbols('m(x) E')
formulas = [
    [(m - E) / (1 + E)],
    [(-m - E) / (1 + E)],
    [(m + E) / (1 + E)]
]

for expr in formulas:
    print(f"[{expr}]")
