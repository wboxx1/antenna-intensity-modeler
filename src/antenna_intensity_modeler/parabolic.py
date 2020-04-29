# -*- coding: utf-8 -*-

"""Main module."""

import numpy as np
import scipy as sp
from scipy import integrate
import scipy.special
import matplotlib.pyplot as plt
import pandas as pd

# Units
m = 1.0
pi = np.pi
cos = np.cos
sin = np.sin
rad = 1.0
s = 1.0


def parameters(radius_meters, freq_mhz, power_watts, efficiency, side_lobe_ratio):
    """Parameters for parabolic dish

    Receives user input parameters for parabolic dish and
    computes and returns all needed parameters for parabolic
    functions.

    :param radius_meters: antenna radius in meters.
    :param freq_mhz: frequency in megahertz.
    :param power_watts: output power of radio in watts.
    :param efficiency: efficiency of antenna.
    :param side_lobe_ratio: side lobe ratio of antenna.
    :type radius_meters: float
    :type freq_mhz: float
    :type power_watts: float
    :type efficiency: float
    :type side_lobe_ratio: float
    :returns: parameters needed for parabolic functions.
    :rtype: tuple(float)
    :Example:

    >>> from antenna_intensity_modeler import parabolic
    >>> params = parabolic.parameters(2.4, 8400., 400.0, 0.62, 20.0)
    >>> params
    (2.4, 8400., 400, 0.62, 20, 0.4872, 1290.24, 2.1134, 175.929)
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


def near_field_corrections(parameters: dict, xbar: float):
    """Near field corrections for parabolic dish.

    Receives user input parameters and normalized off axis distance
    for parabolic dish computes and returns plot of near field correction
    factors.

    :param parameters: parameters tuple created with parameters function
    :param xbar: normalized off-axis distance
    :type parameters: dict
    :type xbar: float
    :returns: dataframe
    :rtype: pandas dataframe
    :Example:

    >>> from antenna_intensity_modeler import parabolic
    >>> import matplotlib.pyplot as plt
    >>> params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    >>> xbar = 1.0
    >>> table = parabolic.near_field_corrections(params, xbar)
    >>> fig, ax = plt.subplots()
    >>> ax.semilogx(table.delta, table.Pcorr)
    >>> ax.set_xlim([0.01, 1.0])
    >>> ax.grid(True, which="both")
    >>> ax.minorticks_on()
    >>> side_lobe_ratio = params[4]
    >>> ax.set_title("Near Field Corrections xbar: %s , slr: %s" % (xbar, side_lobe_ratio))
    >>> ax.set_xlabel("Normalized On Axis Distance")
    >>> ax.set_ylabel("Normalized On Axis Power Density")
    >>> fig.show()

    .. image:: _static/nfcImage.png
    """
    # radius, freq_mhz, power_watts, efficiency, side_lobe_ratio, H, ffmin, ffpwrden, k = tuple(parameters)
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
    delta = np.logspace(-2, 0, 1000)
    Ep = np.zeros(1000)
    count = 0
    xbarR = xbar * radius

    for d in delta:
        theta = np.arctan(xbarR / (d * ffmin))
        u = k * radius * np.sin(theta)

        def fun1(x):
            return (
                1
                * scipy.special.iv(0, pi * H * (1 - x ** 2))  # **(1 / 2))
                # * (1 - x**2)
                * scipy.special.jv(0, u * x)
                * np.cos(pi * x ** 2 / 8 / d)
                * x
            )

        Ep1 = scipy.integrate.romberg(fun1, 0, 1)
        # Ep1 = sum(fun1(np.linspace(0, 1, 1000)))

        def fun2(x):
            return (
                1
                * scipy.special.iv(0, pi * H * (1 - x ** 2))  # **(1 / 2))
                # * (1 - x**2)
                * scipy.special.jv(0, u * x)
                * np.sin(pi * x ** 2 / 8 / d)
                * x
            )

        Ep2 = scipy.integrate.romberg(fun2, 0, 1)
        # Ep2 = sum(fun2(np.linspace(0, 1, 1000)))
        Ep[count] = (1 + np.cos(theta)) / d * abs(Ep1 - 1j * Ep2)
        count += 1

    Pcorr = (Ep ** 2 / Ep[-1] ** 2) * ffpwrden

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


def test_method(parameters, xbar):
    (
        radius,
        freq_mhz,
        power_watts,
        efficiency,
        side_lobe_ratio,
        H,
        ffmin,
        ffpwrden,
        k,
    ) = parameters

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
        print("{} of {}".format(count, len(delta)), end="\r")
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


def hazard_plot(parameters, limit, density=1000, xbar_max=1, gain_boost=None):
    """Hazard plot for parabolic dish.

    Receives user input parameters and hazard limit
    for parabolic dish. Computes and returns hazard distance
    plot.

    :param parameters: parameters dict created with parameters function
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
    delta = np.linspace(1.0, 0.001, n)  # Normalized farfield distances
    # delta = np.logspace(-2, 0, n)
    # delta = delta[::-1]
    xbarArray = np.zeros(n)  # np.ones(n)
    Ep = np.zeros(n)

    # xbars = np.linspace(0, 1, 10)
    step = 0.01
    xbars = np.arange(0, xbar_max + step, step)

    if gain_boost is not None:
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


def print_parameters(parameters):
    """Prints formated parameter list.

    Args:
        parameters(tuple): parameters tuple created with parameters function

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
