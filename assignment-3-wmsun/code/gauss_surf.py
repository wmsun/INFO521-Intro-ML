# gauss_surf.py
# Port of gauss_surf.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Surface and contour plots of a bivariate multivariate Gaussian
import numpy as np
# NOTE: If you are running on MacOSX and encounter an error like the following:
#     RuntimeError: Python is not installed as a framework.
#     The Mac OS X backend will not be able to function correctly
#     if Python is not installed as a framework...
# Then uncomment the following 2 lines:
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D

# Turn off annoying matplotlib warning
import warnings
warnings.filterwarnings("ignore", ".*GUI is implemented.*")

# The Multi-variate Gaussian pdf is given by:
# $p(\mathbf{x}|\mu,\Sigma) =
# \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left\{-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right\}$

# Define the Gaussian parameters
mu = np.array([1, 2])
sigma = np.array([[1, 0.8], [0.8, 4]])

# resolution for plot (larger -> slower rendering)
resolution = 50

# Define the grid for visualisation
X, Y = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))

# Compute the bivariate Gaussian
c1 = 1.0/(2.0 * np.pi)
c2 = 1.0/np.sqrt(np.linalg.det(sigma))
diff = np.concatenate((np.asmatrix(X.flatten(1)).T - mu[0],
                       np.asmatrix(Y.flatten(1)).T - mu[1]),
                      1)
exponent = -0.5 * np.diag(np.dot(np.dot(diff, np.linalg.inv(sigma)), diff.T))
pdfv = c1 * c2 * np.exp(exponent)
pdfv = np.reshape(pdfv, X.shape).T


# Contour plot
plt.figure()
plt.contour(X, Y, pdfv)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, pdfv, rstride=1, cstride=1,
                       cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
fig.colorbar(surf)

plt.pause(.1)  # required on some systems so that rendering can happen

plt.show()

