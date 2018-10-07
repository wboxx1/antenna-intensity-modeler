# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:05:12 2018

@author: wboxx
"""

import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt

# Units
m = 1.0
pi = np.pi
rad = 1.0
s = 1.0


def parameters(radius_meters, freq_mhz, power_watts, efficiency, side_lobe_ratio):
    """Parameters for parabolic dish

    Receives user input parameters for parabolic dish and 
    computes and returns all needed parameters for parabolic
    functions.

    :param radius_meters: antenna radius in meters.
    :param freq_mhz: frequency in hertz.
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

    >>> import parabolic
    >>> params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    >>> params
    (2.4, 8.4e9, 400, 0.62, 20, 0.4872, 1290.24, 2.1134, 175.929)
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
        50: 2.2026
    }
    DIAM = 2 * radius_meters
    LAMDA = C / freq_mhz
    GAIN = 10 * np.log10(efficiency * (pi * DIAM / LAMDA)**2)
    EIRP = power_watts * 10**(0.1 * GAIN)

    """Properties"""
    H = HDICT[side_lobe_ratio]
    ffmin = 2 * DIAM**2 / LAMDA
    ffpwrden = EIRP / (4 * pi * ffmin**2)
    k = 2 * pi / LAMDA

    return radius_meters, freq_mhz, power_watts, efficiency, side_lobe_ratio, H, ffmin, ffpwrden, k


def near_field_corrections(parameters, xbar):
    """Near field corrections for parabolic dish.

    Receives user input parameters and normalized off axis distance
    for parabolic dish computes and returns plot of near field correction
    factors.

    :param parameters: parameters tuple created with parameters function
    :param xbar: normalized off-axis distance
    :type parameters: tuple(float)
    :type xbar: float
    :returns: figure and axes for plot
    :rtype: (figure, axes)
    :Example:
    
    >>> import parabolic
    >>> params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    >>> fig, ax = parabolic.near_field_corrections(params, 1.0)
    """
    radius, freq_mhz, power_watts, efficiency, side_lobe_ratio, H, ffmin, ffpwrden, k = parameters

    delta = np.linspace(0.01, 1.0, 1000)  # Normalized farfield distances
    Ep = np.zeros(1000)
    count = 0
    xbarR = xbar * radius

    for d in delta:
        theta = np.arctan(xbarR / (d * ffmin))
        u = k * radius * np.sin(theta)

        fun1 = lambda x: (sp.special.iv(0, pi * H * (1 - x**2))
                          * sp.special.jv(0, u * x)
                          * np.cos(pi * x**2 / 8 / d)
                          * x)
        Ep1 = scipy.integrate.romberg(fun1, 0, 1)

        fun2 = lambda x: (sp.special.iv(0, pi * H * (1 - x**2))
                          * sp.special.jv(0, u * x)
                          * np.sin(pi * x**2 / 8 / d)
                          * x)
        Ep2 = scipy.integrate.romberg(fun2, 0, 1)
        Ep[count] = (1 + np.cos(theta)) / d * abs(Ep1 - 1j * Ep2)
        count += 1

    Pcorr = (Ep**2 / Ep[-1]**2) * ffpwrden

    fig, ax = plt.subplots()
    ax.semilogx(delta, Pcorr)
    ax.set_xlim([0.01, 1.0])
    ax.grid(True, which="both")
    ax.minorticks_on()
    ax.set_title("Near Field Corrections xbar: %s , slr: %s" % (xbar, side_lobe_ratio))
    ax.set_xlabel("Normalized On Axis Distance")
    ax.set_ylabel("Normalized On Axis Power Density")
    return fig, ax


def hazard_plot(parameters, limit):
    """Hazard plot for parabolic dish.

    Receives user input parameters and hazard limit
    for parabolic dish. Computes and returns hazard distance
    plot.

    :param parameters: parameters tuple created with parameters function
    :param limit: power density limit
    :type parameters: tuple(float)
    :type limit: float
    :returns: figure and axes for hazard plot
    :rtype: (figure, axes)
    :Example:

    >>> import parabolic
    >>> params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    >>> fig, ax = hazard_plot(params, 10.0)
    """
    radius_meters, freq_mhz, power_watts, efficiency, side_lobe_ratio, H, ffmin, ffpwrden, k = parameters
    n = 1000
    delta = np.linspace(1.0, 0.01, n)  # Normalized farfield distances
    xbarArray = np.ones(n)
    xbars = np.linspace(0, 1, 10)
    Ep = np.zeros(1000)

    last = 999
    count = 0
    for d in delta:
        for xbar in xbars:
            xbarR = xbar * radius_meters
            theta = np.arctan(xbarR / (d * ffmin))
            u = k * radius_meters * np.sin(theta)
            fun1 = lambda x: (sp.special.iv(0, pi * H * (1 - x**2))
                              * sp.special.jv(0, u * x)
                              * np.cos(pi * x**2 / 8 / d)
                              * x)
            Ep1 = sp.integrate.romberg(fun1, 0, 1)
            fun2 = lambda x: (sp.special.iv(0, pi * H * (1 - x**2))
                              * sp.special.jv(0, u * x)
                              * np.sin(pi * x**2 / 8 / d)
                              * x)
            Ep2 = sp.integrate.romberg(fun2, 0, 1)
            Ep[count] = (1 + np.cos(theta)) / d * abs(Ep1 - 1j * Ep2)
            power = ffpwrden * (Ep[count]**2 / Ep[0]**2)
            if power - limit < 0:
                if abs(power - limit) < last:
                    xbarArray[count] = xbar
                    last = power - limit
        last = 999
        count += 1

    fig, ax = plt.subplots()
    ax.plot(delta[::-1] * ffmin, xbarArray[::-1] * radius_meters,
            delta[::-1] * ffmin, xbarArray[::-1] * -radius_meters)
    ax.grid(True, which='both')
    ax.minorticks_on()
    ax.set_title('Hazard Plot with limit: %s w/m^2' % limit)
    ax.set_xlabel('Distance From Antenna(m)')
    ax.set_ylabel('Off Axis Distance (m)')
    return fig, ax


def print_parameters(parameters):
    """Prints formated parameter list.

    Args:
        parameters(tuple): parameters tuple created with parameters function

    Returns:
        none
    """
    radius_meters, freq_mhz, power_watts, efficiency, side_lobe_ratio, H, ffmin, ffpwrden, k = parameters
    print('Aperture Radius: %.2f' % radius_meters)
    print('Output Power (w): %.2f' % power_watts)
    print('Antenna Efficiency: %.2f' % efficiency)
    print('Side Lobe Ratio: %.2f' % side_lobe_ratio)
    print('Far Field (m): %.2f' % ffmin)
    print('Far Field (w/m^2): %.2f' % ffpwrden)
