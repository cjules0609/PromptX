# ============================================================================= #
#                  ____  ____   __   _  _  ____  ____  _  _                     #
#                 (  _ \(  _ \ /  \ ( \/ )(  _ \(_  _)( \/ )                    #
#                  ) __/ )   /(  O )/ \/ \ ) __/  )(   )  (                     #
#                 (__)  (__\_) \__/ \_)(_/(__)   (__) (_/\_)                    #
#                                                                               #
# ============================================================================= #
#   PromptX - Prompt X-ray emission modeling of relativistic outflows           #
#   Version 1.0                                                                 #
#   Author: Connery Chen, Yihan Wang, and Bing Zhang                            #
#   License: MIT                                                                #
# ============================================================================= # 

from helper import *
from const import *

class Magnetar:
    """
    Represents a proto-magnetar central engine with spin-down via
    both electromagnetic (EM) and gravitational wave (GW) emission.
    
    Attributes:
        P_0 (float): Initial spin period [s]
        Omega_0 (float): Initial spin frequency [rad/s]
        I (float): Moment of inertia [g cm^2]
        B_p (float): Dipole magnetic field strength [G]
        R (float): Radius of the neutron star [cm]
        eps (float): Ellipticity (dimensionless)
        eta (float): Efficiency factor for energy conversion (unused here)
        eos (dict): Equation of state parameters

        t (np.ndarray): Time evolution [s]
        Omega (np.ndarray): Spin frequency evolution [rad/s]
        tau (np.ndarray): Optical depth over time
        t_tau (float): Time when optical depth = 1
    """

    def __init__(self, collapse=False):
        # Initial spin period and derived angular frequency
        self.P_0 = 1e-3  # [s]
        self.Omega_0 = 2 * np.pi / self.P_0

        # Physical parameters
        self.I = 1e45     # g cm^2
        self.B_p = 1e15   # G
        self.R = 1e6      # cm
        self.eps = 1e-3   # Ellipticity

        # Equation of state parameters
        self.eos = {
            'M_TOV': 2.05,   # Maximum non-rotating NS mass [M_sun]
            'alpha': 1.60,
            'beta': -2.75
        }

        # Spin-down evolution
        self.t, self.Omega = self.spindown(collapse=collapse)

        # Optical depth evolution
        kappa = 1         # Opacity [cm^2/g]
        M_ej = 2e31       # Ejecta mass [g]
        v = 0.3 * c       # Ejecta velocity [cm/s]
        self.tau = kappa * M_ej / (4 * np.pi * v**2 * self.t**2)
        self.t_tau = np.sqrt(M_ej * kappa / (4 * np.pi * v**2))

    def spindown(self, collapse=False):
        """
        Calculates spin-down of a proto-magnetar due to electromagnetic and gravitational wave losses.

        Args:
            collapse (bool): If True, applies a finite collapse time. 
                             Otherwise assumes indefinite spin-down.

        Returns:
            t (np.ndarray): Time array [s]
            Omega (np.ndarray): Angular frequency over time [rad/s]
        """

        # Spindown coefficients: GW (a), EM (b)
        a = (32 * G * self.I * self.eps**2) / (5 * c**5)
        b = (self.B_p**2 * self.R**6) / (6 * c**3 * self.I)

        # Characteristic spin-down timescales
        self.t_0_em = (3 * c**3 * self.I) / (self.B_p**2 * self.R**6 * self.Omega_0**2)
        self.t_0_gw = (5 * c**5) / (128 * G * self.I * self.eps**2 * self.Omega_0**4)

        # Spin frequency array (logarithmic spacing for resolution)
        Omega = np.linspace(self.Omega_0, 10, 1000)

        # Invert spin-down equation to get time as a function of Omega
        t = a / (2 * b**2) * np.log((Omega**2 / self.Omega_0**2) * 
            ((a * self.Omega_0**2 + b) / (a * Omega**2 + b))) + \
            (self.Omega_0**2 - Omega**2) / (2 * b * Omega**2 * self.Omega_0**2)
        
        # Collapse time from EOS
        # M_rem = 2.5
        # P_c = ((M_rem - self.eos['M_TOV']) / 
        #        (self.eos['alpha'] * self.eos['M_TOV']))**(1 / self.eos['beta']) * 1e-3  # [s]
        # Omega_c = 2 * np.pi / P_c
        # self.t_coll = a / (2 * b**2) * np.log((Omega_c**2 / self.Omega_0**2) * 
        #     ((a * self.Omega_0**2 + b) / (a * Omega_c**2 + b))) + \
        #     (self.Omega_0**2 - Omega_c**2) / (2 * b * Omega_c**2 * self.Omega_0**2)

        # Or use default collapse time
        self.t_coll = 300 if collapse else np.inf

        # Clean up negative and NaN values
        t = t[t >= 0]
        t[0] = 1e-6
        Omega = Omega[-len(t):]
        
        return t, Omega