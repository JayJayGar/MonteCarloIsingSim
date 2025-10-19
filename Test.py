import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure
import scipy.ndimage as sp

N = 50

init_random = np.random.random((N,N))
#print(init_random)

lattice_n = np.zeros((N,N))
lattice_n[init_random>=0.25] = 1
lattice_n[init_random<0.25] = -1
#print(lattice_n)

#plt.ion()
#plt.imshow(lattice_n)
#plt.show(block=False)

kern = sp.generate_binary_structure(2,1)

print(kern)