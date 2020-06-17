from scipy import signal
from scipy.fftpack import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt


def my_window(n, N):
    window = np.zeros((N, 1))
    middle = int(N / 2)
    window[middle] = 1
    count = 1
    if N % 2 == 0:
        for a in np.linspace(0, 1, middle):
            amp = (1 - a**2)**n
            window[middle + count - 1] = amp
            window[middle - count] = amp
            count += 1
    else:
        for a in np.linspace(0, 1, middle):
            amp = (1 - a**2)**n
            window[middle + count] = amp
            window[middle - count] = amp
            count += 1
    return window


wind = my_window(1, 51)
plt.plot(wind)
plt.show()

w1D = signal.hamming(51)  # % Some 1D window
plt.plot(w1D)
plt.show()
w1D = w1D.reshape(51,1)
w2D = np.dot(wind[:], wind[:].T)  # % Outer product
print(w2D.shape)
X, Y = np.meshgrid(range(51), range(51))


from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
# fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, intensity, cmap=cm.coolwarm, linewidth=0)
surf = ax.plot_surface(X, Y, w2D, cmap=cm.coolwarm, linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
# cs = ax.contour(X, Y, grid)
# ax.clabel(cs, inline=1, fontsize=-10)
plt.show()
