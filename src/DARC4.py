from antenna_intensity_modeler import parabolic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
from LightPipes import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def au_to_v_per_m(x):
    # return [20*np.log10(x) for x in x]
    return [20*np.log10((x * 5.1422e11)**2 / 377) for x in x]


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


GridSize = 360 * m
GridDimension = 3601
middle = 1800
lambda_ = 0.1 * m  # lambda_ is used because lambda is a Python build-in function.

R = 7.5 * m  # Radius of the aperture

wind = my_window(1, int(2 * R))
w2D = np.dot(wind[:], wind[:].T)  # % Outer product
pad = 1793
w2D = np.pad(w2D, ((pad, pad), (pad, pad)), mode='constant')

field = Begin(GridSize, lambda_, GridDimension)
# print(sum(sum(field)))
# field *= w2D
# field = MultIntensity(w2D, field)
# print(sum(sum(field)))

# field = CircAperture(R, 0, 0, field)

f = 4500 * m
# f1 = 45000 * m
# f2 = f1 * f / (f1 - f)
# field = Lens(f, 0, 0, field)
# field = LensFresnel(f, f, field)
# field = Convert(field)

d = 4500
field = CircAperture(R, 0, 0, field)
# field = Gain(1.0, 1e3, 1e3, field)
field = Lens(f, 0, 0, field)
field = Fresnel(d, field)
# field = Forvard(2250, field)
# field = Forward(2250, field)

# A = lambda_ / 2 / np.pi
# field = Zernike(2, 0, R, A, field)
# field = CircAperture(R, 0, 0, field)

# field = Fresnel(d, field)

intensity = Intensity(0, field)
# intensity = [[x*31.5 for x in row] for row in intensity]
# intensity = [au_to_v_per_m(x) for x in intensity]
# X = range(GridDimension)
# Y = range(GridDimension)
# X, Y = np.meshgrid(X, Y)
# intensity = np.array(intensity)

# fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # surf = ax.plot_surface(X, Y, intensity, cmap=cm.coolwarm, linewidth=0)
# surf = ax.plot_surface(X, Y, pd_field, cmap=cm.coolwarm, linewidth=0)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# # cs = ax.contour(X, Y, grid)
# # ax.clabel(cs, inline=1, fontsize=-10)
# plt.show()

x = []
for i in range(GridDimension):
    x.append((-GridSize * m / 2 + i * GridSize * m / GridDimension) / m)

x = [np.arctan(x / d) * 180 / np.pi for x in x]
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(intensity, cmap='rainbow')
# ax1.axis('off')
# ax2.plot(x, 20*np.log10(intensity[middle]))
ax2.semilogy(x, intensity[middle])
ax2.set_xlabel('x [m]')
ax2.set_ylabel('Intensity [a.u.]')
ax2.grid('on')
plt.show()

# # plt.imshow(intensity)
# # plt.show()
