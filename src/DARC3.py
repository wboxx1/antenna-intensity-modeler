from antenna_intensity_modeler import parabolic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline


# def my_interpolate(x, y, n):
#     x_new = np.linspace(min(x), max(x), n)
#     spl = make_interp_spline(x, y)
#     y_new = spl(x_new)
#     return x_new, y_new


# params = parabolic.parameters(7.5, 3000, 50000, 0.71, 17.57)
# ffmin = params[6]
# C = 1e8
# freq_hz = 3000*1e6
# LAMDA = C / freq_hz

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D
# # fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y, grid = parabolic.combined_hazard_plot(params, 0.1)
# surf = ax.plot_surface(X, Y, grid, cmap=cm.coolwarm, linewidth=0)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# # cs = ax.contour(X, Y, grid)
# # ax.clabel(cs, inline=1, fontsize=-10)
# plt.show()
from LightPipes import *
import matplotlib.pyplot as plt

txr_coord = [
    [-3 * 15, -7],
    [-1 * 15, -7],
    [-1 * 15, -7],
    [1 * 15, -7],
    [3 * 15, -7],
    [3 * 15, -7]
]
GridSize = 150 * m
GridDimension = 150
lambda_ = 0.1 * m  # lambda_ is used because lambda is a Python build-in function.

R = 7.5 * m  # Radius of the aperture
yt = 5 * np.pi / 180  # tilt of the aperture

field = Begin(GridSize, lambda_, GridDimension)
# field = CircScreen(R, txr_coord[0][0], txr_coord[0][1], field)
# field = CircScreen(R, txr_coord[1][0], txr_coord[1][1], field)
# field = CircScreen(R, txr_coord[2][0], txr_coord[2][1], field)
# field = CircScreen(R, txr_coord[3][0], txr_coord[3][1], field)
# field = CircScreen(R, txr_coord[4][0], txr_coord[4][1], field)
# field = CircScreen(R, txr_coord[5][0], txr_coord[5][1], field)

f2 = CircAperture(R, txr_coord[1][0], txr_coord[1][1], field)
f5 = CircAperture(R, txr_coord[4][0], txr_coord[4][1], field)
field1 = BeamMix(f2, f5)
field1 = Tilt(0, -yt, field1)
# field = Fresnel(15, field)

f1 = CircAperture(R, txr_coord[0][0], txr_coord[0][1], field)
f4 = CircAperture(R, txr_coord[3][0], txr_coord[3][1], field)
field2 = BeamMix(f1, f4)
field2 = Tilt(0, -yt, field2)
# field = Fresnel(15, field)

f3 = CircAperture(R, txr_coord[2][0], txr_coord[2][1], field)
f6 = CircAperture(R, txr_coord[5][0], txr_coord[5][1], field)
field3 = BeamMix(f3, f6)
field3 = Tilt(0, -yt, field3)

# field = BeamMix(BeamMix(BeamMix(BeamMix(BeamMix(f1, f2), f3), f4), f5), f6)
# field = Tilt(0, -yt, field)

delta = range(40, 2260, 10)
pd_field = np.zeros((222, GridDimension))
X = range(GridDimension)
Y = range(222)
X, Y = np.meshgrid(X,Y)
count = 0
for d in delta:
    new_field1 = Fresnel(d + 30, field1)
    new_field2 = Fresnel(d + 15, field2)
    new_field3 = Fresnel(d, field3)
    new_field = BeamMix(BeamMix(new_field1, new_field2), new_field3)

    new_intensity = Intensity(1, new_field)
    pd_field[count][:GridDimension] = new_intensity[68][:GridDimension]
    count += 1

# field = Fresnel(2250, field)
# intensity = Intensity(0, field)
# X = range(GridDimension)
# Y = range(GridDimension)
# X, Y = np.meshgrid(X, Y)
# intensity = np.array(intensity)

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
# fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, intensity, cmap=cm.coolwarm, linewidth=0)
surf = ax.plot_surface(X, Y, pd_field, cmap=cm.coolwarm, linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
# cs = ax.contour(X, Y, grid)
# ax.clabel(cs, inline=1, fontsize=-10)
plt.show()

# x = []
# for i in range(GridDimension):
#     x.append((-GridSize * m / 2 + i * GridSize * m / GridDimension) / m)

# fig = plt.figure(figsize=(15, 6))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# ax1.imshow(intensity, cmap='rainbow')
# # ax1.axis('off')
# ax2.plot(x, intensity[128])
# ax2.set_xlabel('x [mm]')
# ax2.set_ylabel('Intensity [a.u.]')
# ax2.grid('on')
# plt.show()

# # plt.imshow(intensity)
# # plt.show()
