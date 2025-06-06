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

from .helper import *
from .const import *
from .magnetar import Magnetar

class Wind(Magnetar):
    """
    Represents a magnetar-powered relativistic wind, with energy and Lorentz factor
    profiles structured as a function of polar angle.
    """

    def __init__(self, n_theta=1000, n_phi=100, g0=50, eps0=1e49, theta_cut=np.pi/2, collapse=False):
        """
        Initializes a wind model for a magnetar-powered outflow 
        and defines its coordinate grid, and energy and Lorentz factor structures

        Parameters:
            n_theta (int): Number of polar (theta) grid points (default is 1000).
            n_phi (int): Number of azimuthal (phi) grid points (default is 100).
            g0 (float): Initial Lorentz factor on the wind axis (default is 50).
            eps0 (float): Initial energy per solid angle (will be overridden by spindown unless code line is commented out).
            theta_cut (float): Cutoff angle for the wind structure (default is pi/2).
            collapse (bool): Flag indicating whether the magnetar undergoes collapse (default is False).
        
        Initializes the following attributes:
            engine (Magnetar): Instance of the Magnetar engine used to model the wind.
            theta_grid (ndarray): 2D grid of theta values for the wind.
            phi_grid (ndarray): 2D grid of phi values for the wind.
            theta (ndarray): 1D array of cell-centered theta values.
            phi (ndarray): 1D array of cell-centered phi values.
            dOmega (ndarray): Differential solid angle for each grid cell.
            theta_cut (float): Cutoff angle for the wind structure.
            eta (float): Conversion efficiency for the magnetar spin-down process.
            eps (ndarray): Energy per solid angle profile of the wind.
            g (ndarray): Lorentz factor profile of the wind.
        """

        # Create a magnetar engine instance
        self.engine = Magnetar(collapse=collapse)

        # Define the bounds for theta (polar angle) and phi (azimuthal angle)
        theta_bounds = [0, np.pi / 2]
        phi_bounds = [0, 2 * np.pi]

        # Generate a grid for theta and phi, based on the specified grid points
        self.theta_grid, self.phi_grid = coord_grid(n_theta, n_phi, theta_bounds, phi_bounds)

        # Compute cell-centered theta and phi values
        self.theta = (self.theta_grid[:-1, :-1] + self.theta_grid[1:, 1:]) / 2
        self.phi = (self.phi_grid[:-1, :-1] + self.phi_grid[1:, 1:]) / 2

        # Compute the differential solid angle (dOmega) for each grid cell
        dtheta = np.gradient(self.theta, axis=1)
        dphi = np.gradient(self.phi, axis=0)
        self.dOmega = np.sin(self.theta) * dtheta * dphi

        # Store cutoff angle for wind structure
        self.theta_cut = theta_cut

        # Compute initial energy per solid angle from magnetar spin-down formula
        self.eta = 0.2   # Conversion efficiency
        eps0 = self.eta * (self.engine.B_p**2 * self.engine.R**6 * self.engine.Omega_0**4) / (6 * c**3) # Comment out if passing user-defined eps0

        # Assign isotropic energy profile
        self.eps = eps_grid(eps0, self.theta, k=0, struct='powerlaw', cutoff=theta_cut)

        # Assign isotropic Lorentz factor profile 
        self.g = gamma_grid(g0, self.theta, k=0, struct='powerlaw', cutoff=theta_cut)

    def observer(self, theta_los=0, phi_los=0, norm=1):
        """
        Compute observer-frame wind emission, including luminosity and spectra.

        Parameters:
            theta_los (float): Observer polar angle (line of sight).
            phi_los (float): Observer azimuthal angle.
            norm (float): Normalization factor (on-axis E_iso of the wind) to scale off-axis E_iso.
        """

        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)

        # Compute Doppler factors
        dopp_on = doppf(self.g, 0)  # On-axis Doppler factor
        dopp_off = doppf(
            self.g,
            angular_d(self.theta[los_coord[1], los_coord[0]], self.theta,
                    self.phi[los_coord[1], los_coord[0]], self.phi)
        )
        self.r_dopp = dopp_off / dopp_on

        # Doppler-boosted energy
        self.eps_prime = self.eps * self.r_dopp
        self.eps_prime3 = self.eps * self.r_dopp**3

        # Patch with maximum Doppler-boosted energy
        # eps_prime_max_index = np.argmax(np.array(self.eps_prime))
        # self.theta_max, self.phi_max = np.unravel_index(eps_prime_max_index, self.eps_prime.shape)

        # Compute effective line-of-sight energy of emitting regions (weighted average)
        eps_cut = self.eps_prime[self.eps_prime != 0]
        r_dopp_cut = self.r_dopp[self.eps_prime != 0]
        dOmega_cut = self.dOmega[self.eps_prime != 0]
        self.eps_prime_los = (np.sum(eps_cut * dOmega_cut * r_dopp_cut**2) / np.sum(dOmega_cut * r_dopp_cut**2))
        # print('Total eps_eff in los:', self.eps_prime_los)

        # Compute base spin-down luminosity
        L_sd = (self.eta * self.engine.B_p**2 * self.engine.R**6 * self.engine.Omega**4) / (6 * c**3)

        # On-axis wind emission
        self.L_on = np.where(self.engine.t > self.engine.t_coll, 0, L_sd)

        # Line-of-sight wind emission
        self.L_los = np.where(
            self.engine.t > self.engine.t_coll, 
            0, 
            np.where(theta_los < self.theta_cut, 
                    L_sd, 
                    L_sd * np.exp(-10 * self.engine.tau)
                )
        )

        # Ratio of Doppler-boosted energy to on-axis energy
        self.r_eps = self.eps_prime_los / norm if self.eps_prime_los > 0 else np.nan

        # Doppler-boosted luminosity
        self.L_dopp = self.L_on * self.r_eps

        # Total observed luminosity
        self.L_X_tot = self.L_los if theta_los < self.theta_cut else self.L_dopp