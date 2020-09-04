import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
import scipy.integrate as integrate

c = 3e8 #m/s
k = 1.38064852e-23
pi = np.pi

class MB:
    def __init__(self, m, T):
        self.m = m
        self.T = T

    def pdf(self, v):
        return np.sqrt(self.m/(2 * pi * k * self.T)) * np.exp(- self.m * v**2/(2 * k * self.T))

    def P(self, v0, v1):
        return integrate.quad(self.pdf, v0, v1)[0]

class Sim:
    def __init__(self):
        self.m = 40 * 1.660539040 * 10**(-27) #kg
        self.T = 2e9 #K
        self.w1 = 2*pi*c/780*10**9 #Hz
        self.w2 =  self.w1 + 2 *pi* 10**9 #Hz
        self.dw1 = 2*pi*30*1e6 #Hz
        self.dw2 = 2*pi*30*1e6 #Hz
        self.rv = MB(self.m, self.T)
        self.N = 1000
        self.default_photons = 1000
        self.nb_photons = self.default_photons
        self.levels = [int(0.5*self.N), int(0.25*self.N), int(0.25*self.N)]

    def saturate(self):
        self.nb_photons = self.default_photons
        self.levels = [int(0.5*self.N), int(0.25*self.N), int(0.25*self.N)]

    def vc(self, wc, wreal, dw, d):
        return np.sign(d)*c*(wc/wreal - 1), d*c*dw/wreal

    def shoot_laser(self, wave):
        #f * lambda = c
        wreal = c/wave
        #for each particle for which wperceived = wc +- dw then there is a probabilty alpha
        #that the photon is absorbed
        # 1 - compute wperceived = wreal(1 +- v/c)
        # 2 - wperceived == wc + dw => wreal(1 +- v/c) == wc + dw
        # newvc == \pm c((wc+dw)/wreal - 1)
        # newvc - oldvc = \pm c((wc + dw)/wreal - 1 - wc/wreal + 1)
        # = \pm c dw / wreal
        # What is the probabilty of having a particle at vc ?
        # P(v= vc) = N*MB.P(vc, vc + dv)*alpha
        for direction in [1, -1]:
            vc1, dv1 = self.vc(self.w1, wreal, self.dw1, direction)
            vc2, dv2 = self.vc(self.w2, wreal, self.dw2, direction)
            p1 = self.rv.P(vc1, vc1+dv1)
            p2 = self.rv.P(vc2, vc2+dv2)
            pexcited1 = self.levels[1]/(self.levels[0] + self.levels[1])
            print(p1)
            pexcited2 = self.levels[2]/(self.levels[0] + self.levels[2])
            self.levels[1] += self.nb_photons*p1*(1 - 2*pexcited1)
            self.levels[0] += self.nb_photons*p1*(2*pexcited1 - 1)
            self.nb_photons += self.nb_photons*p1*(2*pexcited1 - 1)
            self.levels[2] += self.nb_photons*p2*(1 - 2*pexcited2)
            self.levels[0] += self.nb_photons*p2*(2*pexcited2 - 1)
            self.nb_photons += self.nb_photons*p2*(2*pexcited2 - 1)
        return self.nb_photons

    def run_sim(self):
        wavetab = np.linspace(779e-9, 780e-9, 10000)
        photons = []
        for wave in wavetab:
            self.saturate()
            photons.append(self.shoot_laser(wave))
        plt.figure(figsize = (7, 5))
        plt.plot(wavetab, photons)
        plt.show()

s = Sim()
s.run_sim()
