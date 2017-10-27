import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product, count

from matplotlib.colors import LinearSegmentedColormap


# generate uniform unit vectors
def generate_unit_vectors(n):
    'Generates matrix NxN of unit length vectors'
    v = np.random.uniform(-1, 1, (n, n, 2))
    l = np.sqrt(v[:, :, 0] ** 2 + v[:, :, 1] ** 2).reshape(n, n, 1)
    v /= l
    return v


# quintic interpolation
def qz(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


# cubic interpolation
def cz(t):
    return -2 * t * t * t + 3 * t * t


def generate_2D_perlin_noise(size, ns):
    '''
    generate_2D_perlin_noise(size, ns)

    Generate 2D array of size x size filled with Perlin noise.

    Parameters
    ----------
    size : int
        Size of 2D array size x size.
    ns : int
        Distance between nodes.

    Returns
    -------
    m : ndarray
        The 2D array filled with Perlin noise.
    '''
    nc = int(size / ns)  # number of nodes
    grid_size = int(size / ns + 1)  # number of points in grid

    v = generate_unit_vectors(grid_size)

    # generate some constans in advance

    ad, ar = np.arange(ns), np.arange(-ns, 0, 1)

    # vectors from each of 4 nearest nodes to a point in the NSxNS patch
    vd = np.zeros((4, ns, ns, 2))
    for (l1, l2), c in zip(product((ad, ar), repeat=2), count()):
        vd[c, :, :] = np.stack(np.meshgrid(l2, l1, indexing='ij'), axis=2)

    # interpolation coefficients
    d = qz(np.stack((np.zeros((2, ns, ns)),
                     np.stack(np.meshgrid(ad, ad, indexing='ij')))) / ns)
    d[0] = 1 - d[1]

    # make an empy matrix
    m = np.zeros((size, size))
    # reshape for convenience
    t = m.reshape(nc, ns, nc, ns)

    def td(v1, v2):
        return np.tensordot(v1, v2, axes=([0], [2]))

    # fill the image with Perlin noise
    for i, j in product(np.arange(nc), repeat=2):  # loop through the grid
        # calculate values for a NSxNS patch at a time
        n0 = td(v[i, j], vd[0])
        n1 = td(v[i + 1, j], vd[1])
        j0 = d[0, 0] * n0 + d[1, 0] * n1

        n0 = td(v[i, j + 1], vd[2])
        n1 = td(v[i + 1, j + 1], vd[3])
        j1 = d[0, 0] * n0 + d[1, 0] * n1

        t[i, :, j, :] = d[0, 1] * j0 + d[1, 1] * j1

    return m

img = generate_2D_perlin_noise(200, 20)
plt.imshow(img, cmap=cm.gray)

# generate "sky"
# img0 = generate_2D_perlin_noise(400, 80)
# img1 = generate_2D_perlin_noise(400, 40)
# img2 = generate_2D_perlin_noise(400, 20)
# img3 = generate_2D_perlin_noise(400, 10)
#
# img = (img0 + img1 + img2 + img3) / 4
# cmap = LinearSegmentedColormap.from_list('sky',
#                                         [(0, '#0572D1'),
#                                          (0.75, '#E5E8EF'),
#                                          (1, '#FCFCFC')])
# img = cm.ScalarMappable(cmap=cmap).to_rgba(img)
# plt.imshow(img)
