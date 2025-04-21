import numpy as np
import matplotlib.pyplot as plt

from helper import *

def lc(model='pl', t = np.geomspace(1e-3, 1000, 5000), a_1 = 0.5, a_2 = 0.5, c = 1.0):
    if model == 'pl':
        C = np.full(len(t), c)
        F = C * powerlaw(t, a_1) 
    if model == 'fred':
        C = np.full(len(t), c)
        F = C * fred(t, a_1, a_2)
    return t, F, C

def spec(eps, E=np.geomspace(1e2, 1e6, 500), E_det=np.geomspace(1e2, 1e4, 500), alpha=-1, beta=-2.3, E_p=100e3):
    h = 4.14e-15 # ev Hz^-1
    E_0 = E_p / (2 + alpha)
    
    N_E = band(E, alpha, beta, E_0)
    N_E_det = band(E_det, alpha, beta, E_0)

    I_N = 0
    I_EN = 0
    I_EEN = 0

    for i in range(len(E_det) - 1):
        dE = E_det[i+1] - E_det[i]
        mid_N_E = (N_E_det[i] + N_E_det[i+1]) / 2
        I_N += dE * mid_N_E
        I_EN += dE * mid_N_E * dE
        I_EEN += dE * mid_N_E * dE * dE
    A = eps / I_EN
    I_N *= A
    I_EN *= A
    I_EEN *= A
    
    # print('Integral of N(E) | EN(E) | E^2N(E)')
    # print(I_N, ' | ', I_EN, ' | ', I_EEN)
    # print('Normalization Factor A = {}'.format(A))

    return N_E, A
