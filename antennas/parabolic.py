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
m = 1
pi = np.pi
rad = 1
s = 1

class Parabolic:
    def __init__(self, radius, freq, power, efficiency, slr):
        """
        Constructor for parabolic antenna object

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
        C = 3e8*m/s

        # Sidelobe Ratios (illummination)
        # n = 0: slr = 17.57
        # n = 1: slr = 25
        # n = 2: slr = 30
        # n = 3: slr = 35
        # n = 4: slr = 40
        # n = 5: slr = 45
        HDICT = {
                17.57:0,
                20:0.4872,
                25:0.8899,
                30:1.1977,
                35:1.4708,
                40:1.7254,
                45:1.9681,
                50:2.2026
                }
        DIAM = 2*radius
        LAMDA= C/freq
        GAIN = 10*np.log10(efficiency*(pi*DIAM/LAMDA)**2)
        EIRP = power*10**(0.1*GAIN)


        """ Properties """
        self.aperture_radius = radius
        self.power_watts = power
        self.antenna_efficiency = efficiency
        self.side_lobe_ratio = slr
        self.H = HDICT[slr]
        self.ffmin = 2*DIAM**2/LAMDA
        self.ffpwrden = EIRP/(4*pi*self.ffmin**2)
        self.k = 2*pi/LAMDA

#        RANGE_MIN = 0.5*DIAM*(DIAM/LAMDA)**(1/3)
#        if RangeMin > self.ffmin*0.1:
#            print("Rangemin is larger")
#        else:
#            print("0.1*ff is larger")

        print("Parabolic antenna initialized.")
        self.print_stats()

    def update_attributes(self, radius, freq, power, efficiency, slr):
        """
        Updates the attributes of the parabolic antenna instance

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
        C = 3e8*m/s

        # Sidelobe Ratios (illummination)
        # n = 0: slr = 17.57
        # n = 1: slr = 25
        # n = 2: slr = 30
        # n = 3: slr = 35
        # n = 4: slr = 40
        # n = 5: slr = 45
        HDICT = {
                17.57:0,
                20:0.4872,
                25:0.8899,
                30:1.1977,
                35:1.4708,
                40:1.7254,
                45:1.9681,
                50:2.2026
                }
        DIAM = 2*radius
        LAMDA= C/freq
        GAIN = 10*np.log10(efficiency*(pi*DIAM/LAMDA)**2)
        EIRP = power*10**(0.1*GAIN)


        """ Properties """
        self.aperture_radius = radius
        self.power_watts = power
        self.antenna_efficiency = efficiency
        self.side_lobe_ratio = slr
        self.H = HDICT[slr]
        self.ffmin = 2*DIAM**2/LAMDA
        self.ffpwrden = EIRP/(4*pi*self.ffmin**2)
        self.k = 2*pi/LAMDA

        print("Parabolic antenna re-initialized.")
        self.print_stats()

    def near_field_corrections(self, xbar):
        delta = np.linspace(0.01,1,1000) #normalized farfield distances
        Ep = np.zeros(1000)
        count = 0
        xbarR = xbar*self.aperture_radius

        for d in delta:
            theta = np.arctan(xbarR/(d*self.ffmin))
            u = self.k*self.aperture_radius*np.sin(theta)
            fun1 = lambda x: sp.special.iv(0,pi*self.H*(1-x**2))*sp.special.jv(0,u*x)*np.cos(pi*x**2/8/d)*x
            Ep1 = scipy.integrate.romberg(fun1,0,1)
            fun2 = lambda x: sp.special.iv(0,pi*self.H*(1-x**2))*sp.special.jv(0,u*x)*np.sin(pi*x**2/8/d)*x
            Ep2 = scipy.integrate.romberg(fun2,0,1)
            Ep[count] = (1+np.cos(theta))/d*abs(Ep1 - 1j*Ep2)
            count += 1

        Pcorr = (Ep**2/Ep[-1]**2)*self.ffpwrden

        fig, ax = plt.subplots()
        ax.semilogx(delta,Pcorr)
        ax.set_xlim([0.01,1])
        ax.grid(True,which='both')
        ax.minorticks_on()
        ax.set_title("Near Field Corrections xbar: %s , slr: %s" %(xbar, self.side_lobe_ratio))
        ax.set_xlabel("Normalized On Axis Distance")
        ax.set_ylabel("Normalized On Axis Power Density")
        return fig, ax

    def hazard_plot(self, limit):
        n = 1000
        delta = np.linspace(1,0.01,n) #normalized farfield distances
        xbarArray = np.ones(n)
        xbars = np.linspace(0,1,10)
        Ep = np.zeros(1000)

        last = 999
        count = 0
        for d in delta:
            for xbar in xbars:
                xbarR = xbar * self.aperture_radius
                theta = np.arctan(xbarR / (d * self.ffmin))
                u = self.k * self.aperture_radius*np.sin(theta)
                fun1 = lambda x: (sp.special.iv(0, pi * self.H * (1-x**2))
                                  * sp.special.jv(0, u * x)
                                  * np.cos(pi * x**2 / 8 / d)
                                  * x)
                Ep1 = sp.integrate.romberg(fun1, 0, 1)
                fun2 = lambda x: (sp.special.iv(0, pi * self.H * (1 - x**2))
                                  * sp.special.jv(0, u * x)
                                  * np.sin(pi * x**2 / 8 / d)
                                  * x)
                Ep2 = sp.integrate.romberg(fun2, 0, 1)
                Ep[count] = (1 + np.cos(theta)) / d * abs(Ep1 - 1j*Ep2)
                power = self.ffpwrden * (Ep[count]**2 / Ep[0]**2)
                if power - limit < 0:
                    if abs(power - limit) < last:
                        xbarArray[count] = xbar
                        last = power - limit
            last = 999
            count += 1

        fig, ax = plt.subplots()
        ax.plot(delta[::-1] * self.ffmin, xbarArray[::-1] * self.aperture_radius,
                delta[::-1] * self.ffmin, xbarArray[::-1] * -self.aperture_radius)
        #plt.xlim([-2.5,2.5])
        ax.grid(True,which='both')
        ax.minorticks_on()
        ax.set_title('Hazard Plot with limit: %s w/m^2' %limit)
        ax.set_xlabel('Distance From Antenna(m)')
        ax.set_ylabel('Off Axis Distance (m)')
        return fig, ax

    def print_stats(self):
        print('Aperture Radius: %.2f' %self.aperture_radius)
        print('Output Power (w): %.2f' %self.power_watts)
        print('Antenna Efficiency: %.2f' %self.antenna_efficiency)
        print('Side Lobe Ratio: %.2f' %self.side_lobe_ratio)
        print('Far Field (m): %.2f' %self.ffmin)
        print('Far Field (w/m^2): %.2f' %self.ffpwrden)