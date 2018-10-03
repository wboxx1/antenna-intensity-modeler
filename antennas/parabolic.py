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


def parameters(radius, freq, power, efficiency, slr):
    """
    Set parameters for parabolic dish
        inputs:
                radius - antenna radius in meters
                freq - frequency in hertz
                power - output power of radio in watts
                efficiency - efficiency of antenna
                slr - side lobe ratio of antenna

        output:
            none
    """

    """ Constants """
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
    DIAM = 2 * radius
    LAMDA = C / freq
    GAIN = 10 * np.log10(efficiency * (pi * DIAM / LAMDA)**2)
    EIRP = power * 10**(0.1 * GAIN)

    """ Properties """
    H = HDICT[slr]
    ffmin = 2 * DIAM**2 / LAMDA
    ffpwrden = EIRP / (4 * pi * ffmin**2)
    k = 2 * pi / LAMDA

    return radius, freq, power, efficiency, slr, H, ffmin, ffpwrden, k


def near_field_corrections(parameters, xbar):
    radius, freq, power, efficiency, slr, H, ffmin, ffpwrden, k = parameters

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
    ax.grid(True, which='both')
    ax.minorticks_on()
    ax.set_title("Near Field Corrections xbar: %s , slr: %s" % (xbar, slr))
    ax.set_xlabel("Normalized On Axis Distance")
    ax.set_ylabel("Normalized On Axis Power Density")
    return fig, ax


def hazard_plot(parameters, limit):
    radius, freq, power, efficiency, slr, H, ffmin, ffpwrden, k = parameters
    n = 1000
    delta = np.linspace(1.0, 0.01, n)  # Normalized farfield distances
    xbarArray = np.ones(n)
    xbars = np.linspace(0, 1, 10)
    Ep = np.zeros(1000)

    last = 999
    count = 0
    for d in delta:
        for xbar in xbars:
            xbarR = xbar * radius
            theta = np.arctan(xbarR / (d * ffmin))
            u = k * radius * np.sin(theta)
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
    ax.plot(delta[::-1] * ffmin, xbarArray[::-1] * radius,
            delta[::-1] * ffmin, xbarArray[::-1] * -radius)
    ax.grid(True, which='both')
    ax.minorticks_on()
    ax.set_title('Hazard Plot with limit: %s w/m^2' % limit)
    ax.set_xlabel('Distance From Antenna(m)')
    ax.set_ylabel('Off Axis Distance (m)')
    return fig, ax


def print_parameters(parameters):
    radius, freq, power, efficiency, slr, H, ffmin, ffpwrden, k = parameters
    print('Aperture Radius: %.2f' % radius)
    print('Output Power (w): %.2f' % power)
    print('Antenna Efficiency: %.2f' % efficiency)
    print('Side Lobe Ratio: %.2f' % slr)
    print('Far Field (m): %.2f' % ffmin)
    print('Far Field (w/m^2): %.2f' % ffpwrden)
