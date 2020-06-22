from antenna_intensity_modeler import parabolic
from antenna_intensity_modeler.helpers import Either, Option
import matplotlib.pyplot as plt
import time
import warnings
from functools import partial
from typing import Optional, Union
import numpy as np


def nfc_intermediary(xbar, params, res=1000):
    return parabolic.near_field_corrections(
        parameters=params, xbar=xbar, resolution=res
    )


def match(monad: Union[Either, Option], some_fun, no_fun):
    if isinstance(monad, Some):
        return monad.flat_map(some_fun)
    elif isinstance(monad, Right):
        return monad.flat_map(some_fun)
    else:
        return no_fun(monad.value)


def plotting_intermediary(nfc, delta, ax):
    # _nfc = match(nfc, lambda x: x.pure, lambda x: [])
    _nfc = nfc
    return ax.semilogx(delta, _nfc)


# warnings.filterwarnings("ignore")
params = parabolic.parameters(2.4, 8400, 400.0, 0.62, 17.57)
xbar = np.arange(0, 1.1, 0.1)
# xbar = [1.1, 1.2, 1.5, 2.0]
resolution = 1000
corrections_with_params = partial(nfc_intermediary, params=params, res=resolution)
nfc = list(map(corrections_with_params, xbar))
delta = np.logspace(-2, 0, resolution)

fig, ax = plt.subplots()
ax.set_xlim([0.01, 1.0])
ax.grid(True, which="both")
ax.minorticks_on()
slr = params.get("side_lobe_ratio")
ax.set_title("Near Field Corrections xbar: {}, slr: {}".format(xbar, slr))
ax.set_xlabel("Normalized On Axis Distance")
ax.set_ylabel("Normalized On Axis Power Density")

partial_log_plot = partial(plotting_intermediary, delta=delta, ax=ax)
_ = list(map(partial_log_plot, nfc))
plt.show()

# limit = 10.0
# print("Run 1...", end=" ", flush=True)
# t1 = time.clock()
# df = parabolic.test_hazard_plot(params, limit, xbar_max=2.0)
# t2 = time.clock()
# print("done in {:.2f} seconds".format(t2 - t1), flush=True)
# # print("Run 2...", end=" ", flush=True)
# # t3 = time.clock()
# # df2 = parabolic.hazard_plot(params, limit)
# # t4 = time.clock()
# # print("done in {:.2f} seconds".format(t4 - t3), flush=True)

# rng = df.range.values
# positives = df.positives.values
# negatives = df.negatives.values
# fig, ax = plt.subplots()
# ax.plot(rng, positives, rng, negatives)
# ax.grid(True, which="both")
# ax.minorticks_on()
# ax.set_title("Hazard Plot with limit: %s w/m^2" % limit)
# ax.set_xlabel("Distance From Antenna(m)")
# ax.set_ylabel("Off Axis Distance (m)")
# plt.show()
