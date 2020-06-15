from antenna_intensity_modeler import parabolic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline


def my_interpolate(x, y, n):
    x_new = np.linspace(min(x), max(x), n)
    spl = make_interp_spline(x, y)
    y_new = spl(x_new)
    return x_new, y_new


params = parabolic.parameters(7.5, 3000, 312500, 0.71, 17.57)
ffmin = params[6]
C = 1e8
freq_hz = 3000*1e6
LAMDA = C / freq_hz

fig, ax = plt.subplots()
_density = 100
el_axis = 9
degree = 5
rotate = [
    [np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180)],
    [-np.sin(degree * np.pi / 180), np.cos(degree * np.pi / 180)]
]

# plot off axis limitation
x = np.linspace(0.01, 1.0, _density)*ffmin
y = []
thetas = []
for x_bar in x:
    limit = 0.
    theta = 0.
    while limit < 0.08:
        limit = (np.pi * 7.5**2) / (LAMDA * x_bar) * np.sin(theta * np.pi / 180.)**2
        theta += 0.5
    y.append(x_bar * np.tan(theta * np.pi / 180))
    thetas.append(theta)

x, y = my_interpolate(x, y, _density * 10)
y_neg = [-x for x in y]
# ax.plot(x, y, x, y_neg)

# rotate
top = np.dot(np.column_stack((x, y)), rotate)
bottom = np.dot(np.column_stack((x, y_neg)), rotate)

# add elevation
top[:, 1] += el_axis
bottom[:, 1] += el_axis

# plot
ax.plot(top[:, 0], top[:, 1], bottom[:, 0], bottom[:, 1])
# limitation_lines = pd.DataFrame(
#     np.column_stack((top[:, 0], top[:, 1], bottom[:, 0], bottom[:, 1])),
#     columns=['top_x', 'top_y', 'bottom_x', 'bottom_y']
# )
# limitation_lines.to_csv('limitation_line_angle_{}.csv'.format(degree))

# penis plot
limits = [0.9]#, 38.2, 14.9, 128.4]
# limits = [10., 100.]
for limit in limits:
    table = parabolic.hazard_plot(params, limit, xbar_max=15, density=_density, gain_boost=6)
    _x = table['range']
    _y = table['positives']
    # _y_neg = table['negatives']
    # smooth
    _x, _y = my_interpolate(_x, _y, _density * 10)
    _y_neg = [-x for x in _y]
    # rotate
    top = np.dot(np.column_stack((_x, _y)), rotate)
    bottom = np.dot(np.column_stack((_x, _y_neg)), rotate)
    # add elevation
    top[:, 1] += el_axis
    bottom[:, 1] += el_axis
    # plot
    ax.plot(top[:, 0], top[:, 1], bottom[:, 0], bottom[:, 1])
    limit_lines = pd.DataFrame(
        np.column_stack((top[:, 0], top[:, 1], bottom[:, 0], bottom[:, 1])),
        columns=[
            '{}_top_x'.format(limit),
            '{}_top_y'.format(limit),
            '{}_bottom_x'.format(limit),
            '{}_bottom_y'.format(limit)]
    )
    limit_lines.to_csv('limit_line_{}_angle_{}.csv'.format(limit, degree))

ax.grid(True, which='both')
ax.minorticks_on()
# ax.set_title('Hazard Plot with limit: %s w/m^2' % limit)
ax.set_xlabel('Distance From Antenna(m)')
ax.set_ylabel('Off Axis Distance (m)')
ax.set_xlim([0, 2250])
ax.set_ylim([-40, 40])
# plot elevation axis
ax.plot(x, np.ones(len(x)) * el_axis)

# fig.show()
plt.show()

# D = 15
# # xbars = [0*D, D/3, 2*D/3, D, 4*D/3, 5*D/3, 2*D, 7*D/3, 8*D/3]
# # xbars = np.arange(0,3.1,0.1)
# xbars = [0]
# fig, ax = plt.subplots()
# ax.set_xlim([0.01, 1.0])
# ax.grid(True, which='both')
# ax.minorticks_on()
# ax.set_title('Relative Field Intensity')
# ax.set_xlabel('Normalized Z-axis distance')
# ax.set_ylabel('Relative Field Strength in dB')
# standard = 0
# for xbar in xbars:
#     table = parabolic.near_field_corrections(params, xbar)
#     if xbar == 0:
#         standard = table.Pcorr[0]
#     ax.semilogx(table.delta, table.Pcorr)
# fig.show()

# D = 15
# fig, ax = plt.subplots()
# ax.set_xlim([0.01, 1.0])
# ax.grid(True, which='both')
# ax.minorticks_on()
# ax.set_title('Relative Field Intensity')
# ax.set_xlabel('Normalized Z-axis distance')
# ax.set_ylabel('Relative Field Strength in dB')
# standard = 0
# table = parabolic.test_method(params, 0)
# # for xbar in xbars:
# #     table = parabolic.near_field_corrections(params, xbar)
# #     if xbar == 0:
# #         standard = table.Pcorr[0]
# #     ax.semilogx(table.delta, table.Pcorr)
# ax.semilogx(table.delta, table.Pcorr)
# plt.show()

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