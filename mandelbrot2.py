import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis


def compute_mandelbrot(n_max, threshold, nx, ny):
    # A grid of c-values
    # x = np.linspace(-2, 1, nx)
    # y = np.linspace(-1.5, 1.5, ny)
    x = np.linspace(0.2, 1, nx)
    y = np.linspace(-0.5, 0.5, ny)

    c = x[:, newaxis] + 1j*y[newaxis, :]

    # z = c
    z = np.zeros((nx, ny), dtype=complex)
    M = np.full((nx, ny), True, dtype=bool)
    N = np.zeros((nx, ny))

    for j in range(n_max):
        z[M] = z[M] * z[M] + c[M]
        M[np.abs(z) > threshold] = False
        N[M] = j

    return N


mb_set = compute_mandelbrot(50, 2., 3000, 3000)

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
ax.set_xticks([])
ax.set_yticks([])
plt.imshow(np.rot90(mb_set), cmap='hot')
plt.savefig('mandelbrot.png')
plt.close()

