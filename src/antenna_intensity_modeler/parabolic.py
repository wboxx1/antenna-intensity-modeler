# -*- coding: utf-8 -*-

"""Main module for parabolic reflector antennas."""

import numpy as np
import scipy as sp
import scipy.special
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from multiprocessing import Pool
from typing import Union, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# from .helpers import Either, Left, Right
from pymonad.tools import curry
from pymonad.operators.either import Either, Left, Right
from pymonad.operators.maybe import Maybe, Nothing, Just
from pymonad.operators.list import ListMonad

# Basic units
METER = 1.0
HERTZ = 1.0
SECOND = 1.0
PI = np.pi
COS = np.cos
SIN = np.sin
RAD = 1.0
DEG = PI / 180
TO_DEG = 180. / PI

# Other units
GHZ = 1e9 * HERTZ
MHZ = 1e6 * HERTZ
KM = 1e3 * METER

# Math Constants
C = 299792458 * METER / SECOND


def parameters(
    radius_meters: float,
    freq_mhz: float,
    power_watts: float,
    efficiency: float,
    side_lobe_ratio: float,
    a_value: float = None,
    n_value: float = None,
    gain: float = None,
) -> dict:
    """Parameters for parabolic dish

    Receives user input parameters for parabolic dish and
    computes and returns all needed parameters for parabolic
    functions.

    Args:
        radius_meters (float): antenna radius in meters.
        freq_mhz (float): frequency in megahertz.
        power_watts (float): output power of radio in watts.
        efficiency (float): efficiency of antenna.
        side_lobe_ratio (float): side lobe ratio of antenna.

    Returns:
        dict: parameter dictionary needed for parabolic functions.

    Example:
        >>> from antenna_intensity_modeler import parabolic
        >>> params = parabolic.parameters(2.4, 8400., 400.0, 0.62, 20.0)
        >>> params
        {'radius_meters': 2.4, 'freq_mhz': 8400.0, 'power_watts': 400.0,
        'efficiency': 0.62, 'side_lobe_ratio':20.0, 'H': 0.4872,
        'ffmin': 1290.24, 'ffpwrden': 2.1134, 'k': 175.929}
    """

    """Constants"""

    freq_hz = freq_mhz * MHZ
    diam = 2 * radius_meters
    _lamda = C / freq_hz
    if gain is None:
        gain = 10 * np.log10(efficiency * (PI * diam / _lamda) ** 2)
    eirp = power_watts * 10 ** (0.1 * gain)

    # Check aperture size limitation
    if diam < 10 * _lamda:
        ratio = diam / _lamda
        print(
            (
                "WARNING - Diameter should be at least ten times "
                +
                "the wavelength for this method to work.  Currently "
                +
                "the diameter is {} times the wavelength."
            ).format(ratio)
        )

    """Properties"""
    # h = hdict[side_lobe_ratio]
    # First find h given side lobe ratio
    # using the following relationship
    # slr = 17.57 dB + 20 * log10(( 2 * I_1(pi * H)) / (pi * H))
    hansons_fun = (
        lambda x: side_lobe_ratio
        - 17.57
        - 20 * np.log10((2 * scipy.special.i1(PI * x)) / (PI * x))
    )
    h_value = scipy.optimize.newton(hansons_fun, 0.1)

    ffmin = 2 * diam**2 / _lamda
    ffpwrden = eirp / (4 * PI * ffmin**2)
    k = 2 * PI / _lamda

    min_range = 0.5 * diam * (diam / _lamda)**(1/3)

    return_dict = {
        "radius_meters": radius_meters,
        "freq_mhz": freq_mhz,
        "lamda": _lamda,
        "power_watts": power_watts,
        "efficiency": efficiency,
        "side_lobe_ratio": side_lobe_ratio,
        "H": h_value,
        "ffmin": ffmin,
        "ffpwrden": ffpwrden,
        "k": k,
        "a_value": a_value,
        "n_value": n_value,
        "min_range": min_range
    }
    return return_dict


def _bessel_func(
    x: float, f: Union[np.cos, np.sin], H: float, u: float, d: float
) -> Callable:
    return (
        1.
        * scipy.special.i0(PI * H * (1. - x ** 2))  # **(1 / 2))
        # * (1 - x**2)
        * scipy.special.j0(u * x)
        * f(PI * x ** 2 / 8. / d)
        * x
    )


def _romberg_integration(
    fun: Callable, lower: int = 0, upper: int = 1, divmax: int = 20
) -> Either:
    try:
        return Right(scipy.integrate.romberg(fun, lower, upper, divmax=divmax))
    except Exception as err:
        return Left(err)


def _check_theta(theta, a, z, lamda):
    if (PI * a**2 / lamda / z) * SIN(theta)**2 > 0.08:
        print(
            (
                "WARNING - Failed theta check for theta = {}, "
                +
                "a = {}, and z = {}"
            ).format(theta * TO_DEG, a, z)
        )


def _run_near_field_corrections(
    d: float,
    parameters: dict,
    xbar: float
) -> float:
    # Get parameters
    radius = parameters.get("radius_meters")
    H = parameters.get("H")
    ffmin = parameters.get("ffmin")
    k = parameters.get("k")
    lamda = parameters.get("lamda")

    xbarR = xbar * radius
    theta = np.arctan(xbarR / (d * ffmin))
    _check_theta(theta, xbarR, d*ffmin, lamda)
    u = k * radius * np.sin(theta)

    # Get Bessel Functions
    bessel_func_cos = partial(_bessel_func, f=np.cos, H=H, u=u, d=d)
    bessel_func_sin = partial(_bessel_func, f=np.sin, H=H, u=u, d=d)

    # Calculate Powers
    Ep1 = _romberg_integration(bessel_func_cos, 0, 1, divmax=20)
    Ep2 = _romberg_integration(bessel_func_sin, 0, 1, divmax=20)
    # Ep1_1 = scipy.integrate.romberg(bessel_func_cos, 0, 1, divmax=20)
    # Ep2_1 = scipy.integrate.romberg(bessel_func_sin, 0, 1, divmax=20)

    @curry(2)
    def final_reduction(x, y):
        return (1. + np.cos(theta)) / d * abs(x - 1.j * y)

    # return (1 + np.cos(theta)) / d * abs(Ep1_1 - 1j * Ep2_1)
    return (final_reduction << Ep1 & Ep2).then(_square)


def _square(x):
    return Just(x ** 2)


def _squared(x):
    return _square << x


@curry(2)
def _divide(x, y):
    return Just(x / y)


@curry(2)
def _normalize(y, x):
    if y == 0:
        return Nothing
    else:
        return x / y


def _unpack(x):
    return x.value


def near_field_corrections(
    parameters: dict, xbar: float, resolution: int = 1000, use_light_pipes: bool = False
) -> np.array:
    """Near field corrections for parabolic dish.

    Receives user input parameters and normalized off axis distance
    (tangental from plane of propagation) from parabolic dish center.
    Computes and returns the near field correction factors.

    Args:
        parameters (dict): parameters dictionary created with parameters
        function.
        xbar (float)  : normalized off-axis distance from dish center.
        resolution (float): number of points used in array.

    Returns:
        numpy.Array: An array with axial correction factors

    Example:
        >>> from antenna_intensity_modeler import parabolic
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> params = parabolic.parameters(2.4, 8400, 400.0, 0.62, 20.0)
        >>> xbar = 1.0
        >>> resolution = 1000
        >>> power_norm = parabolic.near_field_corrections(params, xbar, resolution)
        >>> fig, ax = plt.subplots()
        >>> delta = np.logspace(-2, 0, resolution)
        >>> ax.semilogx(delta, power_norm)
        >>> ax.set_xlim([0.01, 1.0])
        >>> ax.grid(True, which="both")
        >>> ax.minorticks_on()
        >>> slr = params.get('side_lobe_ratio')
        >>> ax.set_title("Near Field Corrections xbar: {}, slr: {}".format(xbar, slr))
        >>> ax.set_xlabel("Normalized On Axis Distance")
        >>> ax.set_ylabel("Normalized On Axis Power Density")
        >>> plt.show()

    .. image:: _static/nfcImage.png
    """

    run_with_params = partial(
        _run_near_field_corrections, parameters=parameters, xbar=xbar
    )

    min_range = parameters.get("min_range")
    ffmin = parameters.get("ffmin")

    if min_range < 0.1 * ffmin:
        my_func = (lambda x: min_range - (10**x * ffmin))
        pow = scipy.optimize.newton(my_func, -2)
    else:
        pow = -2

    delta = np.logspace(-2, 0, resolution)

    def run(f, my_iter):
        iter_length = len(my_iter)
        with tqdm(total=iter_length) as pbar:
            # let's give it some more threads:
            with ProcessPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(f, arg): arg for arg in my_iter}
                results = {}
                for future in as_completed(futures):
                    arg = futures[future]
                    results[arg] = future.result()
                    pbar.update(1)
        return results

    Ep = []

    Ep_dict = run(run_with_params, delta)
    for key in sorted(Ep_dict.keys()):
        Ep.append(Ep_dict[key])

    ff_val = Ep[-1].value
    Ep_monad = ListMonad(*Ep)

    power_norm = Ep_monad * _unpack * _normalize(ff_val)

    # power_norm = Ep ** 2 / Ep[-1] ** 2  # * parameters["ffpwrden"]
    return power_norm


def delta_xbar_split(delta_xbar: tuple, parameters: dict):
    (d, xbar) = delta_xbar[0], delta_xbar[1]
    return _run_near_field_corrections(d, parameters, xbar)


def get_normalized_power_tensor(
    parameters: dict, density: int = 1000, xbar_max: float = 1.0
) -> np.array:
    n = density
    delta = np.linspace(0.001, 1.0, num=n)  # Normalized farfield distances

    # step = 0.01
    step = xbar_max / density * 10
    xbars = np.arange(0, xbar_max, step)

    delta_xbars = np.reshape(np.array(np.meshgrid(delta, xbars)).T, (-1, 2))

    chunksize = 100
    run_corrections_with_params = partial(delta_xbar_split, parameters=parameters)
    with ProcessPoolExecutor() as p:
        # map each delta, xbars tuple to the run_with_params partial function
        mtrx = np.array(
            list(
                p.map(
                    run_corrections_with_params,
                    delta_xbars,
                    chunksize=chunksize
                )
            )
        )
    # Reshape the resulting flattened array into a 2-d tensor representing
    # 2-d space from txr center to farfield and down to txr edge
    mtrx_reshaped = np.reshape(mtrx, (len(xbars), len(delta)), order="F")
    # Normalize each row of tensor to maximum level at far field
    return mtrx_reshaped ** 2 / mtrx_reshaped[:, -1][:, np.newaxis] ** 2


def test_hazard_plot(
    parameters: dict,
    limit: float,
    density: int = 1000,
    xbar_max: float = 1.0,
    gain_boost_db: int = 0.0,
) -> pd.DataFrame:
    """Hazard plot for parabolic dish.

    Receives user input parameters and hazard limit
    for parabolic dish. Computes and returns hazard distance
    plot.

    Args:
        parameters (dict): parameters dict created with parameters function
        limit (float): power density limit
        density (int): (optional) number of points for plot, if none density=1000
        xbar_max (float): (optional) maximum value for xbar, if none is given xbar_max=1
        gain_boost (int): (optional) additional gain in dB to add to output power

    Returns:
        pandas.DataFrame: dataframe containing a range column, a positive values column, and a negative values column

    Example:
        >>> from antenna_intensity_modeler import parabolic
        >>> import matplotlib.pyplot as plt
        >>>
        >>> params = parabolic.parameters(2.4, 8400, 400.0, 0.62, 20.0)
        >>> limit = 10.0
        >>> df = parabolic.hazard_plot(params, limit)
        >>> rng = df.range.values
        >>> positives = df.positives.values
        >>> negatives = df.negatives.values
        >>>
        >>> fig, ax = plt.subplots()
        >>> ax.plot(range, positives, range, negatives)
        >>> ax.grid(True, which='both')
        >>> ax.minorticks_on()
        >>> ax.set_title('Hazard Plot with limit: %s w/m^2' % limit)
        >>> ax.set_xlabel('Distance From Antenna(m)')
        >>> ax.set_ylabel('Off Axis Distance (m)')

    .. image:: _static/hazard_plot.png
    """
    radius_meters = parameters.get("radius_meters")
    ffpwrden = parameters.get("ffpwrden")
    ffmin = parameters.get("ffmin")

    gain_boost = 10 ** (gain_boost_db / 10.0)
    ffpwrden_boosted = gain_boost * ffpwrden
    delta = np.linspace(0.001, 1.0, num=density)  # Normalized farfield distances
    xbar_density = (xbar_max + 0.01) / 0.01

    # Get the normalized power tensor
    mtrx_normalized = get_normalized_power_tensor(
        parameters, density=density, xbar_max=xbar_max
    )

    # Subtract normalized limit from normalized power tensor
    mtrx_limited = mtrx_normalized - limit / ffpwrden_boosted

    pause = True
    # find the index of the largest values less than zero
    # divide by xbar density to determine position
    mtrx_arg_max = np.argmax(
        np.where(mtrx_limited < 0, mtrx_limited, -np.inf), axis=0
    ) * (xbar_max / density * 10)

    # mtrx_reduced = np.maximum.reduce(
    #     mtrx_limited, initial=-np.inf, where=[mtrx_limited < 0]
    # )
    return pd.DataFrame(
        dict(
            range=delta * ffmin,
            positives=mtrx_arg_max * radius_meters,
            negatives=mtrx_arg_max * -radius_meters,
        )
    )


def hazard_plot(
    parameters: dict,
    limit: float,
    density: int = 1000,
    xbar_max: float = 1.0,
    gain_boost_db: int = 0,
):
    """Hazard plot for parabolic dish.

    Receives user input parameters and hazard limit
    for parabolic dish. Computes and returns hazard distance
    plot.

    Args:
        parameters (dict): parameters dict created with parameters function
        limit (float): power density limit
        density (int): (optional) number of points for plot, if none density=1000
        xbar_max (float): (optional) maximum value for xbar, if none is given xbar_max=1
        gain_boost (int): (optional) additional gain in dB to add to output power

    Returns:
        pandas.DataFrame: dataframe containing a range column, a positive values column, and a negative values column

    Example:
        >>> from antenna_intensity_modeler import parabolic
        >>> import matplotlib.pyplot as plt
        >>>
        >>> params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
        >>> limit = 10.0
        >>> df = parabolic.hazard_plot(params, limit)
        >>> rng = df.range.values
        >>> positives = df.positives.values
        >>> negatives = df.negatives.values
        >>>
        >>> fig, ax = plt.subplots()
        >>> ax.plot(range, positives, range, negatives)
        >>> ax.grid(True, which='both')
        >>> ax.minorticks_on()
        >>> ax.set_title('Hazard Plot with limit: %s w/m^2' % limit)
        >>> ax.set_xlabel('Distance From Antenna(m)')
        >>> ax.set_ylabel('Off Axis Distance (m)')

    .. image:: _static/hazard_plot.png
    """

    radius_meters = parameters.get("radius_meters")
    freq_mhz = parameters.get("freq_mhz")
    power_watts = parameters.get("power_watts")
    efficiency = parameters.get("efficiency")
    side_lobe_ratio = parameters.get("side_lobe_ratio")
    H = parameters.get("H")
    ffmin = parameters.get("ffmin")
    ffpwrden = parameters.get("ffpwrden")
    k = parameters.get("k")

    n = density
    delta = np.linspace(1.0, 0.001, n)  # Normalized farfield distances
    # delta = np.logspace(-2, 0, n)
    # delta = delta[::-1]
    xbarArray = np.zeros(n)  # np.ones(n)
    Ep = np.zeros(n)

    # xbars = np.linspace(0, 1, 10)
    step = 0.01
    xbars = np.arange(0, xbar_max + step, step)

    gain_boost = 10 ** (gain_boost_db / 10.0)
    ffpwrden = gain_boost * ffpwrden

    last = 1e-9
    count = 0
    ff_reference = 0

    for d in delta:
        for xbar in xbars:
            xbarR = xbar * radius_meters
            theta = np.arctan(xbarR / (d * ffmin))
            u = k * radius_meters * np.sin(theta)

            def fun1(x):
                return (
                    1
                    # * sp.special.iv(0, PI * H * (1 - x**2)**(1 / 2))
                    * sp.special.jv(0, u * x)
                    * np.cos(PI * x ** 2 / 8 / d)
                    * x
                )

            # Ep1 = sp.integrate.romberg(fun1, 0, 1)
            Ep1 = sum(fun1(np.linspace(0, 1, 1000)))

            def fun2(x):
                return (
                    1
                    # * sp.special.iv(0, PI * H * (1 - x**2)**(1 / 2))
                    * sp.special.jv(0, u * x)
                    * np.sin(PI * x ** 2 / 8 / d)
                    * x
                )

            # Ep2 = sp.integrate.romberg(fun2, 0, 1)
            Ep2 = sum(fun2(np.linspace(0, 1, 1000)))

            if d == 1.0 and xbar == 0.0:
                ff_reference = (1 + np.cos(theta)) / d * abs(Ep1 - 1j * Ep2)
                Ep = ff_reference
            else:
                Ep = (1 + np.cos(theta)) / d * abs(Ep1 - 1j * Ep2)

            power = ffpwrden * (Ep ** 2 / ff_reference ** 2)
            if (power - limit) > 0:
                # if power - limit > last:
                xbarArray[count] = xbar
                # last = power - limit
        last = 1e-9
        count += 1

    # fig, ax = plt.subplots()
    # # ax.plot(delta[::-1] * ffmin, xbarArray[::-1] * radius_meters,
    # #         delta[::-1] * ffmin, xbarArray[::-1] * -radius_meters)
    # _x = delta[::-1] * ffmin
    # _y = xbarArray[::-1] * radius_meters
    # _y_neg = -1 * _y
    # degree = 0
    # rotate = [
    #     [np.cos(degree * np.PI / 180), np.sin(degree * np.PI / 180)],
    #     [-np.sin(degree * np.PI / 180), np.cos(degree * np.PI / 180)]
    # ]
    # top = np.dot(np.column_stack((_x, _y)), rotate)
    # bottom = np.dot(np.column_stack((_x, _y_neg)), rotate)
    # ax.plot(top[:, 0], top[:, 1], bottom[:, 0], bottom[:, 1])
    # ax.grid(True, which='both')
    # ax.minorticks_on()
    # ax.set_title('Hazard Plot with limit: %s w/m^2' % limit)
    # ax.set_xlabel('Distance From Antenna(m)')
    # ax.set_ylabel('Off Axis Distance (m)')
    # return fig, ax
    return pd.DataFrame(
        dict(
            range=delta[::-1] * ffmin,
            positives=xbarArray[::-1] * radius_meters,
            negatives=xbarArray[::-1] * -radius_meters,
        )
    )


def combined_hazard_plot(parameters, limit, density=1000):
    """Hazard plot for parabolic dish.

    Receives user input parameters and hazard limit
    for parabolic dish. Computes and returns hazard distance
    plot.

    :param parameters: parameters tuple created with parameters function
    :param limit: power density limit
    :param density: (optional) number of points for plot, if none density=1000
    :param xbar_max: (optional) maximum value for xbar, if none is given xbar_max=1
    :param gain_boost: (optional) additional numerical gain to add
    :type parameters: tuple(float)
    :type limit: float
    :type xbar_max: int
    :type gain_boost: float
    :returns: figure and axes for hazard plot
    :rtype: (figure, axes)
    :Example:

    >>> from antenna_intensity_modeler import parabolic
    >>> params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    >>> fig, ax = hazard_plot(params, 10.0)
    """
    radius_meters = parameters.get("radius_meters")
    freq_mhz = parameters.get("freq_mhz")
    power_watts = parameters.get("power_watts")
    efficiency = parameters.get("efficiency")
    side_lobe_ratio = parameters.get("side_lobe_ratio")
    H = parameters.get("H")
    ffmin = parameters.get("ffmin")
    ffpwrden = parameters.get("ffpwrden")
    k = parameters.get("k")

    n = density
    delta = np.linspace(1.0, 0.01, n)  # Normalized farfield distances
    # x = np.logspace(-5, 0, 100)  # Normalized farfield distances
    x = np.linspace(0, 1, 1000)  # Normalized to farfield distance
    y = np.arange(0, 16.1, 0.1)  # Normalized to radius
    X, Y = np.meshgrid(x, y)

    txr_coord = [
        [-30 / ffmin, 2],
        [-45 / ffmin, 6],
        [-15 / ffmin, 6],
        [-30 / ffmin, 10],
        [-45 / ffmin, 14],
        [-15 / ffmin, 14],
    ]

    pd_field = np.zeros(X.shape)
    eps = 1e-12

    # Determine farfield power density
    d = 1.0
    theta = 0.0
    u = 0.0

    def fun1(x):
        return (
            1
            # * scipy.special.iv(0, PI * H * (1 - x**2))
            * scipy.special.jv(0, u * x)
            * np.cos(PI * x ** 2 / 8 / d)
            * x
        )

    Ep1 = scipy.integrate.romberg(fun1, 0, 1, divmax=20)
    # Ep1 = sum(fun1(np.linspace(0, 1, 1000)))

    def fun2(x):
        return (
            1
            # * scipy.special.iv(0, PI * H * (1 - x**2))
            * scipy.special.jv(0, u * x)
            * np.sin(PI * x ** 2 / 8 / d)
            * x
        )

    Ep2 = scipy.integrate.romberg(fun2, 0, 1, divmax=20)
    # Ep2 = sum(fun2(np.linspace(0, 1, 1000)))

    ff_reference = (1 + np.cos(theta)) / d * abs(Ep1 - 1j * Ep2)

    for i in range(len(y)):
        for j in range(len(x)):
            for coord in txr_coord:
                _x = coord[0]
                _y = coord[1]
                diff_x = np.abs(X[i, j] - _x) * ffmin
                diff_y = np.abs(Y[i, j] - _y) * radius_meters
                theta = np.arctan(diff_y / (diff_x + eps))
                u = k * radius_meters * np.sin(theta)

                def fun1(x):
                    return (
                        1
                        # * scipy.special.iv(0, PI * H * (1 - x**2))
                        * scipy.special.jv(0, u * x)
                        * np.cos(PI * x ** 2 / 8 / abs(X[i, j] - _x))
                        * x
                    )

                Ep1 = scipy.integrate.romberg(fun1, 0, 1, divmax=20)
                # Ep1 = sum(fun1(np.linspace(0, 1, 1000)))

                def fun2(x):
                    return (
                        1
                        # * scipy.special.iv(0, PI * H * (1 - x**2))
                        * scipy.special.jv(0, u * x)
                        * np.sin(PI * x ** 2 / 8 / abs(X[i, j] - _x))
                        * x
                    )

                Ep2 = scipy.integrate.romberg(fun2, 0, 1, divmax=20)
                # Ep2 = sum(fun2(np.linspace(0, 1, 1000)))

                Ep = (1 + np.cos(theta)) / abs(X[i, j] - _x) * abs(Ep1 - 1j * Ep2)
                pd_field[i, j] = (
                    pd_field[i, j] + (Ep ** 2 / ff_reference ** 2) * ffpwrden
                )

                # Pcorr = (Ep**2 / Ep[-1]**2) * ffpwrden

    return X, Y, pd_field


def _run_far_field_radiation_pattern(theta: float, parameters: dict) -> float:
    # Get parameters
    radius = parameters.get("radius_meters")
    H = parameters.get("H")
    k = parameters.get("k")

    u = k * radius * np.sin(theta)

    # Get Bessel Functions
    bessel_func_cos = partial(_bessel_func, f=np.cos, H=H, u=u, d=1)
    bessel_func_sin = partial(_bessel_func, f=np.sin, H=H, u=u, d=1)

    # Calculate Powers
    Ep1 = _romberg_integration(bessel_func_cos, 0, 1, divmax=20)
    Ep2 = _romberg_integration(bessel_func_sin, 0, 1, divmax=20)

    @curry(2)
    def final_reduction(x, y):
        return (1 + np.cos(theta)) * abs(x - 1j * y)

    # return (1 + np.cos(theta)) / d * abs(Ep1_1 - 1j * Ep2_1)
    return (final_reduction << Ep1 & Ep2).then(_square)


def far_field_radiation_pattern(parameters, theta, N):
    """Far field radiation pattern for parabolic dish.

    Receives user input parameters, the angle in radians the user wants
    the pattern for, and the number of points required in the plot.
    Computes and returns the normalized power values for the far field
    radiation pattern.

    Args:
        parameters (dict): parameters dictionary created from the parameters
        function.
        theta (float)  : angle in radians to indicate edge of pattern.
        N (float): number of points used in array.

    Returns:
        numpy.Array: An array that contains the one sided normalized array
        pattern

    Example:
        >>> from antenna_intensity_modeler import parabolic
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> params = parameters(2.4, 8400, 400.0, 0.62, 20.0)
        >>> theta = 4.0 * np.pi / 180
        >>> resolution = 1024
        >>> power_norm = far_field_radiation_pattern(params, theta, resolution)
        >>> fig, ax = plt.subplots()
        >>> theta_array = np.linspace(0, theta * 180 / np.pi, resolution)
        >>> ax.plot(theta_array, 10 * np.log10(power_norm))
        >>> ax.set_xlim([0.0, 4.0])
        >>> ax.grid(True, which="both")
        >>> ax.minorticks_on()
        >>> slr = params.get('side_lobe_ratio')
        >>> ax.set_title("Far Field Radiation Pattern, slr: {}".format(slr))
        >>> ax.set_xlabel("Angle Off Boresight (deg)")
        >>> ax.set_ylabel("Normalized Power Density (dB)")
        >>> plt.show()

    .. image:: _static/ffPattern.png
    """

    run_with_params = partial(
        _run_far_field_radiation_pattern, parameters=parameters
    )

    def run(f, my_iter):
        iter_length = len(my_iter)
        with tqdm(total=iter_length) as pbar:
            # let's give it some more threads:
            with ProcessPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(f, arg): arg for arg in my_iter}
                results = {}
                for future in as_completed(futures):
                    arg = futures[future]
                    results[arg] = future.result()
                    pbar.update(1)
        return results

    angles = np.linspace(0, theta, N)
    Ep_dict = run(run_with_params, angles)
    Ep = []
    for key in sorted(Ep_dict.keys()):
        Ep.append(Ep_dict[key])

    ff_val = Ep[0].value
    Ep_monad = ListMonad(*Ep)

    power_norm = Ep_monad * _unpack * _normalize(ff_val)

    return power_norm


def print_parameters(parameters: dict):
    """Prints formated parameter list.

    Args:
        parameters (dict): parameters tuple created with parameters function
    Returns:
        none
    """
    radius_meters = parameters.get("radius_meters")
    power_watts = parameters.get("power_watts")
    efficiency = parameters.get("efficiency")
    side_lobe_ratio = parameters.get("side_lobe_ratio")
    ffmin = parameters.get("ffmin")
    ffpwrden = parameters.get("ffpwrden")

    print("Aperture Radius: %.2f" % radius_meters)
    print("Output Power (w): %.2f" % power_watts)
    print("Antenna Efficiency: %.2f" % efficiency)
    print("Side Lobe Ratio: %.2f" % side_lobe_ratio)
    print("Far Field (m): %.2f" % ffmin)
    print("Far Field (w/m^2): %.2f" % ffpwrden)


def _next_pow_2(x):
    return 1 << (int(x) - 1).bit_length()


if __name__ == "__main__":
    
    params = parameters(2.4, 8400, 400.0, 0.62, 20.0)
    limit = 10.0
    df = test_hazard_plot(params, limit)
    rng = df.range.values
    positives = df.positives.values
    negatives = df.negatives.values
    fig, ax = plt.subplots()
    ax.plot(rng, positives, rng, negatives)
    ax.grid(True, which='both')
    ax.minorticks_on()
    ax.set_title('Hazard Plot with limit: %s w/m^2' % limit)
    ax.set_xlabel('Distance From Antenna(m)')
    ax.set_ylabel('Off Axis Distance (m)')

    freq = 44.5 * GHZ
    lamda = C / freq
    R = 6.5 * METER
    D = 2 * R
    P0 = 130

    params = parameters(
        radius_meters=R,
        freq_mhz=freq / MHZ,
        power_watts=P0,
        efficiency=0.62,
        side_lobe_ratio=35,
    )

    xbar = 0.0
    npix = 1024
    length = 100
    size = int(_next_pow_2(10 * D))
    resolution = size / npix

    theta = 1 * DEG
    power_norm = far_field_radiation_pattern(params, theta, npix)
    fig, ax = plt.subplots()
    delta = params.get("k") * R * SIN(np.linspace(0, 1 * DEG, npix))
    # ax.set_xlim([0, 12])
    # ax.set_ylim([-40, 0])
    ax.plot(delta, 10*np.log10(power_norm))

    # power_norm = near_field_corrections(params, xbar, length)
    # delta = np.logspace(-2, 0, npix)
    # ax.semilogx(delta, 10*np.log10(power_norm))
    # ax.set_xlim([0.01, 1.0])

    ax.grid(True, which="both")
    ax.minorticks_on()
    slr = params.get("side_lobe_ratio")
    ax.set_title("Near Field Corrections xbar: {}, slr: {}".format(xbar, slr))
    ax.set_xlabel("Normalized On Axis Distance")
    ax.set_ylabel("Normalized On Axis Power Density")
    plt.show()
