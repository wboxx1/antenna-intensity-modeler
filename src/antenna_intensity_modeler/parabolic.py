# -*- coding: utf-8 -*-

"""Main module for parabolic reflector antennas."""

import numpy as np
import scipy as sp
from scipy import integrate
import scipy.special
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from multiprocessing import Pool
from typing import Union, Callable
from concurrent.futures import ProcessPoolExecutor

# from .helpers import Either, Left, Right
import pymonad
# from pymonad import *
from pymonad.operators import *
from pymonad.operators import Either
# import pymonad.tools
curry = pymonad.tools.curry
# from .local_imports.pymonad import *
# from .local_imports.pymonad.tools import curry
# from .local_imports.pymonad.operators import *

# Units
m = 1.0
pi = np.pi
cos = np.cos
sin = np.sin
rad = 1.0
s = 1.0


def parameters(
    radius_meters: float,
    freq_mhz: float,
    power_watts: float,
    efficiency: float,
    side_lobe_ratio: float,
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
        {'radius_meters': 2.4, 'freq_mhz': 8400.0, 'power_watts': 400.0, 'efficiency': 0.62,
        'side_lobe_ratio':20.0, 'H': 0.4872, 'ffmin': 1290.24, 'ffpwrden': 2.1134, 'k': 175.929}
    """

    """Constants"""
    C = 3e8 * m / s

    # Sidelobe Ratios (illummination)
    # n = 0: slr = 17.57
    # n = 1: slr = 25
    # n = 2: slr = 30
    # n = 3: slr = 35
    # n = 4: slr = 40
    # n = 5: slr = 45
    HDICT = {
        17.57: 0,
        20: 0.4872,
        25: 0.8899,
        30: 1.1977,
        35: 1.4708,
        40: 1.7254,
        45: 1.9681,
        50: 2.2026,
    }
    freq_hz = freq_mhz * 1e6
    DIAM = 2 * radius_meters
    LAMDA = C / freq_hz
    GAIN = 10 * np.log10(efficiency * (pi * DIAM / LAMDA) ** 2)
    EIRP = power_watts * 10 ** (0.1 * GAIN)

    """Properties"""
    H = HDICT[side_lobe_ratio]
    ffmin = 2 * DIAM ** 2 / LAMDA
    ffpwrden = EIRP / (4 * pi * ffmin ** 2)
    k = 2 * pi / LAMDA

    return_dict = {
        "radius_meters": radius_meters,
        "freq_mhz": freq_mhz,
        "power_watts": power_watts,
        "efficiency": efficiency,
        "side_lobe_ratio": side_lobe_ratio,
        "H": H,
        "ffmin": ffmin,
        "ffpwrden": ffpwrden,
        "k": k,
    }
    return return_dict
    # return radius_meters, freq_mhz, power_watts, efficiency, side_lobe_ratio, H, ffmin, ffpwrden, k


def bessel_func(
    x: float, f: Union[np.cos, np.sin], H: float, u: float, d: float
) -> Callable:
    return (
        1
        * scipy.special.iv(0, pi * H * (1 - x ** 2))  # **(1 / 2))
        # * (1 - x**2)
        * scipy.special.jv(0, u * x)
        * f(pi * x ** 2 / 8 / d)
        * x
    )


@curry
def bessel_func_test(
    H: float, u: float, d: float, f: Union[np.cos, np.sin], x: float
) -> Callable:
    return (
        1
        * scipy.special.iv(0, pi * H * (1 - x ** 2))  # **(1 / 2))
        # * (1 - x**2)
        * scipy.special.jv(0, u * x)
        * f(pi * x ** 2 / 8 / d)
        * x
    )


def romberg_integration(
    fun: Callable, lower: int = 0, upper: int = 1, divmax: int = 20
) -> Either:
    try:
        return Right(scipy.integrate.romberg(fun, lower, upper, divmax=divmax))
    except Exception as err:
        return Left(err)


def run_near_field_corrections(d: float, parameters: dict, xbar: float) -> float:
    # Get parameters
    radius = parameters.get("radius_meters")
    freq_mhz = parameters.get("freq_mhz")
    power_watts = parameters.get("power_watts")
    efficiency = parameters.get("efficiency")
    side_lobe_ratio = parameters.get("side_lobe_ratio")
    H = parameters.get("H")
    ffmin = parameters.get("ffmin")
    ffpwrden = parameters.get("ffpwrden")
    k = parameters.get("k")

    xbarR = xbar * radius
    theta = np.arctan(xbarR / (d * ffmin))
    u = k * radius * np.sin(theta)

    # Get Bessel Functions
    bessel_func_cos = partial(bessel_func, f=np.cos, H=H, u=u, d=d)
    bessel_func_sin = partial(bessel_func, f=np.sin, H=H, u=u, d=d)

    # _bessel_func = bessel_func_test(H, u, d)
    # bessel_func_cos = _bessel_func(np.cos)
    # bessel_func_sin = _bessel_func(np.sin)

    # Calculate Powers
    Ep1 = romberg_integration(bessel_func_cos, 0, 1, divmax=20)
    Ep2 = romberg_integration(bessel_func_sin, 0, 1, divmax=20)
    # Ep1 = scipy.integrate.romberg(bessel_func_cos, 0, 1, divmax=20)
    # Ep2 = scipy.integrate.romberg(bessel_func_sin, 0, 1, divmax=20)

    @curry
    def final_reduction(x, y):
        return (1 + np.cos(theta)) / d * abs(x - 1j * y)

    return final_reduction * Ep1 & Ep2
    # return (1 + np.cos(theta)) / d * abs(Ep1 - 1j * Ep2)


def near_field_corrections(
    parameters: dict, xbar: float, resolution: int = 1000
) -> np.array:
    """Near field corrections for parabolic dish.

    Receives user input parameters and normalized off axis distance
    for parabolic dish.  Computes and returns plot of near field correction
    factors.

    Args:
        parameters (tuple): parameters tuple created with parameters function.
        xbar (float)  : normalized off-axis distance.
        resolution (float): number of points used in array.

    Returns:
        pandas.DataFrame: A dataframe with "delta" and "Pcorr" columns

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
        run_near_field_corrections, parameters=parameters, xbar=xbar
    )
    delta = np.logspace(-2, 0, resolution)
    # with ProcessPoolExecutor() as p:
    #     Ep = np.array(list(p.map(run_with_params, delta)))
    # power_norm = Ep ** 2 / Ep[-1] ** 2  # * parameters["ffpwrden"]
    # Ep = list(map(run_with_params, delta))
    Eplist = list(map(run_with_params, delta))
    Ep = ListMonad(Eplist)
    power_norm = Ep ** 2 / Ep[-1] ** 2  # * parameters["ffpwrden"]
    @curry
    def unwrap(init_val, x):
        return x.getValue() ** 2 / init_val.getValue() ** 2
    unwrap_with_init = unwrap(Ep[-1])
    # power_norm = list(map(unwrap_with_init, Ep))

    # return pd.DataFrame(dict(delta=delta, Pcorr=Pcorr))
    return power_norm


def test_method(parameters, xbar):
    radius = parameters.get("radius_meters")
    freq_mhz = parameters.get("freq_mhz")
    power_watts = parameters.get("power_watts")
    efficiency = parameters.get("efficiency")
    side_lobe_ratio = parameters.get("side_lobe_ratio")
    H = parameters.get("H")
    ffmin = parameters.get("ffmin")
    ffpwrden = parameters.get("ffpwrden")
    k = parameters.get("k")

    # delta = np.linspace(0.01, 1.0, 1000)  # Normalized farfield distances
    # delta = np.logspace(-2, 0, 100)
    delta = range(70)
    Ep = np.zeros(len(delta))
    count = 0
    xbarR = xbar * radius
    C = 3e8
    lamda = C / (freq_mhz * 1e6)
    # theta = np.arctan(xbarR / (delta * ffmin))
    # sintheta = np.sin(theta)
    # x = np.linspace(-7.5, 7.5, 1000)
    # u = k * radius * np.sin(theta)
    # Ep = np.fft.fft(
    #     1
    #     # * scipy.special.iv(0, pi * H * (1 - x**2))
    #     # * scipy.special.jv(0, u * x)
    #     * np.exp(1j * k * (x**2 / 2 / xbarR))
    # )
    # Ep = np.fft.fftshift(Ep)
    # Ep = Ep * np.exp(-1j * k * 7.5**2 * u**2 / 2 / xbarR)

    # Pcorr = abs(Ep)
    # for d in delta:

    ###########################3
    # FSA method without dropping phi

    # delta = np.logspace(-2, 0, 1000)
    # Ep = np.zeros(len(delta))
    # for d in delta:
    #     theta = np.arctan(xbarR / (d * ffmin))
    #     u = k * radius * np.sin(theta)

    #     def fun1(x,y): return(
    #         1
    #         # * scipy.special.iv(0, pi * H * (1 - x**2))
    #         * np.exp(-1j * k * x**2 / 2 / d)
    #         * np.exp(1j * k * x * sin(theta) * cos((pi / 8) - y) * x)
    #     )
    #     Ep1 = integrate.dblquad(fun1, 0, radius, lambda x: 0, lambda x: 2 * pi)

    #     # def fun2(x,y): return(
    #     #     1
    #     #     # * scipy.special.iv(0, pi * H * (1 - x**2))
    #     #     * sin(k * x**2 / 2 / d)
    #     #     * sin(k * x * sin(theta) * cos((pi / 8) - y) * x)
    #     # )
    #     # Ep2 = integrate.dblquad(fun1, 0, radius, lambda x: 0, lambda x: 2 * pi)

    #     # Ep[count] = (1 + cos(theta)) / 4 / pi / d * abs(Ep1[0] - 1j * Ep2[0])
    #     Ep[count] = (1 + cos(theta)) / 4 / pi / d * abs(Ep1[0])
    #     count += 1
    ##########################

    ##########################
    # Hansen implementation
    d = 0.075 * ffmin
    for deg in delta:
        # theta = np.arctan(xbarR / (d * ffmin))
        # u = np.sin(theta)
        theta = deg * pi / 180
        phi = 0
        # R = (xbar**2 + d**2)**(1/2)
        R = d / cos(theta)  # / ffmin
        nabla = 120 * pi

        # u = k * radius * np.sin(theta)

        #######################
        # o = k * xbar**2 / d

        # def fun1(x): return(
        #     1
        #     * np.exp(1j * k * x**2 / 2 / d)
        #     * np.exp(-1j * k * x * u)
        # )
        # def fun1(x): return (
        #     1
        #     * scipy.special.iv(0, pi * H * (1 - x**2))
        #     # * scipy.special.jv(0, u * x)
        #     * np.cos(k * (o / 2) * x**2 * (1 - u**2))
        #     * np.cos(k * u * x)
        # )j
        # Ep[count] = 13.14 * (1 - np.cos(np.pi / 8 / d))
        #######################

        ##############################
        # def R(x,y): return(
        #     ((xbarR - (x * cos(y)))**2 + (-x * sin(y))**2 + d**2)**1/2
        # )

        # def fun1(x,y): return(
        #     1
        #     * scipy.special.iv(0, pi * H * (1 - x**2))
        #     * (1 / R(x,y))
        #     * cos(k * R(x,y))
        # )
        # Ep1 = integrate.dblquad(fun1, 0, radius, lambda x: 0, lambda x: 2 * pi)

        # def fun2(x,y): return(
        #     1
        #     * scipy.special.iv(0, pi * H * (1 - x**2))
        #     * (1 / R(x,y))
        #     * sin(k * R(x,y))
        # )
        # Ep2 = integrate.dblquad(fun2, 0, radius, lambda x: 0, lambda x: 2 * pi)

        # Ep[count] = (1 / lamda) * abs(Ep1[0] - 1j * Ep2[0])**2
        ############################

        #######################
        summation = 0
        even = 0
        odd = 0
        # print(k)
        # print(radius)
        # print(H)

        # print("{} of {}".format(count, len(delta)), end="\r")
        for N in range(50):

            def fun_b1(x):
                return (
                    1
                    # * scipy.special.iv(0, pi * H * (1 - x**2)**(1 / 2))
                    * scipy.special.spherical_jn(N, x)
                    * x
                )

            B1 = scipy.integrate.romberg(fun_b1, 0, k * radius, divmax=20)

            def fun_b2(x):
                return (
                    1
                    # * scipy.special.iv(0, pi * H * (1 - x**2)**(1 / 2))
                    * scipy.special.spherical_jn(N, x)
                )

            # B2 = scipy.integrate.romberg(fun_b2, 0, 1, divmax=20)
            B2 = scipy.integrate.romberg(fun_b2, 0, k * radius, divmax=20)

            if N % 2 == 0:

                def fun_a1(x):
                    return scipy.special.gegenbauer(N, 0)(sin(theta) * cos(x))

                def fun_a2(x):
                    return scipy.special.gegenbauer(N - 1, 1)(sin(theta) * cos(x))

                def fun_a5(x):
                    return (
                        scipy.special.gegenbauer(N - 2, 2)(sin(theta) * cos(x))
                        * cos(x) ** 2
                    )

                if N == 0:
                    A1 = 0
                    A2 = 0
                    A5 = 0
                    pass
                elif N == 2:
                    A1 = scipy.integrate.romberg(fun_a1, 0, pi / 2, divmax=20)
                    A2 = scipy.integrate.romberg(fun_a2, 0, pi / 2, divmax=20)
                    A5 = 0
                else:
                    A1 = scipy.integrate.romberg(fun_a1, 0, pi / 2, divmax=20)
                    A2 = scipy.integrate.romberg(fun_a2, 0, pi / 2, divmax=20)
                    A5 = scipy.integrate.romberg(fun_a5, 0, pi / 2, divmax=20)

                even += (
                    (2 * N + 1)
                    # * scipy.special.spherical_yn(N, k * R)
                    * scipy.special.hankel2(N, k * R)
                    * (
                        -A1 * B1
                        + ((1 / k / R) * A2 * B2)
                        - ((1 / k ** 2 / R ** 2) * A5 * B1)
                    )
                )
            else:

                def fun_a4(x):
                    return scipy.special.gegenbauer(N - 2, 2)(
                        sin(theta) * cos(x)
                    ) * cos(x)

                if N == 1:
                    A4 = 0
                else:
                    A4 = scipy.integrate.romberg(fun_a4, 0, pi / 2, divmax=20)

                odd += (
                    (2 * N + 1)
                    # * scipy.special.spherical_yn(N, k * R)
                    * scipy.special.hankel2(N, k * R)
                    * A4
                    * B2
                )
        even = nabla * cos(theta) * cos(phi) / pi * even
        odd = nabla * sin(theta) * cos(theta) * cos(phi) / pi / k / R * odd
        # print(even)
        # print(odd)
        Ep[count] = abs(
            np.real(even) + np.real(odd) + 1j * (np.imag(even) + np.imag(odd))
        )
        # print(Ep[count])

        # def fun2(x): return (
        #     1
        #     * scipy.special.iv(0, pi * H * (1 - x**2))
        #     # * scipy.special.jv(0, u * x)
        #     * np.sin(k * (o / 2) * x**2 * (1 - u**2))
        #     * np.sin(k * u * x)
        # )
        # Ep2 = 2 * scipy.integrate.romberg(fun2, 0, 1)
        # Ep2 = sum(fun2(np.linspace(0, 1, 1000)))
        # Ep[count] = abs(Ep1 - 1j * Ep2)
        # Ep2 = np.exp(-1j * k * radius**2 * u**2 / 2 / d)

        count += 1

    Pcorr = Ep ** 2 / max(Ep) ** 2  # * ffpwrden

    # Pcorr = (Ep**2 / Ep[-1]**2) * ffpwrden
    # Pcorr = Ep

    # fig, ax = plt.subplots()
    # ax.semilogx(delta, Pcorr)
    # ax.set_xlim([0.01, 1.0])
    # ax.grid(True, which="both")
    # ax.minorticks_on()
    # ax.set_title("Near Field Corrections xbar: %s , slr: %s" % (xbar, side_lobe_ratio))
    # ax.set_xlabel("Normalized On Axis Distance")
    # ax.set_ylabel("Normalized On Axis Power Density")
    # return fig, ax
    return pd.DataFrame(dict(delta=delta, Pcorr=Pcorr))


def delta_xbar_split(delta_xbar: tuple, parameters: dict):
    (d, xbar) = delta_xbar[0], delta_xbar[1]
    return run_near_field_corrections(d, parameters, xbar)


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
            list(p.map(run_corrections_with_params, delta_xbars, chunksize=chunksize))
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
                    # * sp.special.iv(0, pi * H * (1 - x**2)**(1 / 2))
                    * sp.special.jv(0, u * x)
                    * np.cos(pi * x ** 2 / 8 / d)
                    * x
                )

            # Ep1 = sp.integrate.romberg(fun1, 0, 1)
            Ep1 = sum(fun1(np.linspace(0, 1, 1000)))

            def fun2(x):
                return (
                    1
                    # * sp.special.iv(0, pi * H * (1 - x**2)**(1 / 2))
                    * sp.special.jv(0, u * x)
                    * np.sin(pi * x ** 2 / 8 / d)
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
    #     [np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180)],
    #     [-np.sin(degree * np.pi / 180), np.cos(degree * np.pi / 180)]
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
            # * scipy.special.iv(0, pi * H * (1 - x**2))
            * scipy.special.jv(0, u * x)
            * np.cos(pi * x ** 2 / 8 / d)
            * x
        )

    Ep1 = scipy.integrate.romberg(fun1, 0, 1, divmax=20)
    # Ep1 = sum(fun1(np.linspace(0, 1, 1000)))

    def fun2(x):
        return (
            1
            # * scipy.special.iv(0, pi * H * (1 - x**2))
            * scipy.special.jv(0, u * x)
            * np.sin(pi * x ** 2 / 8 / d)
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
                        # * scipy.special.iv(0, pi * H * (1 - x**2))
                        * scipy.special.jv(0, u * x)
                        * np.cos(pi * x ** 2 / 8 / abs(X[i, j] - _x))
                        * x
                    )

                Ep1 = scipy.integrate.romberg(fun1, 0, 1, divmax=20)
                # Ep1 = sum(fun1(np.linspace(0, 1, 1000)))

                def fun2(x):
                    return (
                        1
                        # * scipy.special.iv(0, pi * H * (1 - x**2))
                        * scipy.special.jv(0, u * x)
                        * np.sin(pi * x ** 2 / 8 / abs(X[i, j] - _x))
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


def print_parameters(parameters: dict):
    """Prints formated parameter list.

    Args:
        parameters (dict): parameters tuple created with parameters function
    Returns:
        none
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
    print("Aperture Radius: %.2f" % radius_meters)
    print("Output Power (w): %.2f" % power_watts)
    print("Antenna Efficiency: %.2f" % efficiency)
    print("Side Lobe Ratio: %.2f" % side_lobe_ratio)
    print("Far Field (m): %.2f" % ffmin)
    print("Far Field (w/m^2): %.2f" % ffpwrden)
