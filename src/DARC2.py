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


params = parabolic.parameters(7.5, 3000, 50000, 0.71, 17.57)
# params = parabolic.parameters(7.5, 3000, 50000, 0.71, 25)
ffmin = params[6]
C = 1e8
freq_hz = 3000*1e6
LAMDA = C / freq_hz

fig, ax = plt.subplots()
# ax.set_xlim([0.01, 1.0])
ax.grid(True, which='both')
ax.minorticks_on()
ax.set_title('Relative Field Intensity')
ax.set_xlabel('Normalized Z-axis distance')
ax.set_ylabel('Relative Field Strength in dB')
standard = 0
table = parabolic.test_method(params, 0)
ax.plot(table.delta, table.Pcorr)
# xbars = np.arange(0,1.1,0.1)
# for xbar in xbars:
#     table = parabolic.near_field_corrections(params, xbar)
#     if xbar == 0:
#         standard = table.Pcorr[0]
#     ax.semilogx(table.delta, table.Pcorr / params[7] )
plt.show()