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

    def __init__(self, n_theta=1000, n_phi=100, L=1e48, g0=50, L0=1e48, theta_cut=np.pi/2, k=0, collapse=False, wind_struct=1):
        """
        Initializes a wind model for a magnetar-powered outflow 
        and defines its coordinate grid, and energy and Lorentz factor structures

        Parameters:
            n_theta (int): Number of polar (theta) grid points (default is 1000).
            n_phi (int): Number of azimuthal (phi) grid points (default is 100).
            g0 (float): Initial Lorentz factor on the wind axis (default is 50).
            L0 (float): Initial luminosity per solid angle (will be overridden by spindown unless code line is commented out).
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
        theta_bounds = [0, np.pi]
        phi_bounds = [0, 2 * np.pi]

        # Generate a grid for theta and phi, based on the specified grid points
        self.theta_grid, self.phi_grid = coord_grid(n_theta, n_phi, theta_bounds, phi_bounds)

        # Compute cell-centered theta and phi values
        self.theta = (self.theta_grid[:-1, :-1] + self.theta_grid[1:, 1:]) / 2
        self.phi = (self.phi_grid[:-1, :-1] + self.phi_grid[1:, 1:]) / 2

        # Compute the differential solid angle (dOmega) for each grid cell
        dtheta = np.gradient(self.theta, axis=0)
        dphi = np.gradient(self.phi, axis=1)
        self.dOmega = np.sin(self.theta) * dtheta * dphi

        # Store core angle for wind structure (used in Gaussian/PL profile)
        self.theta_c = np.radians(20)

        self.k = k  # only used for PL profile

        # Store cutoff angle for wind structure
        self.theta_cut = theta_cut

        # Compute initial energy per solid angle from magnetar spin-down formula
        self.eta = 0.1   # Conversion efficiency
        self.define_structure(g0, L0, wind_struct)
        self.normalize(L0)

    def define_structure(self, g0, L0, wind_struct):
        """
        Defines the structure of the wind's energy and Lorentz factor profiles based on the specified profile type.

        Parameters:
        g0 (float): The initial Lorentz factor normalization.
        L0 (float): The initial luminosity per solid angle (before applying structure).
        E_iso (float): The isotropic-equivalent energy used for the Gaussian and power-law structures.
        struct (int or function): The structure type, where:
            - 1: Tophat
            - 2: Gaussian
            - 3: Power-law
            - function: A custom piecewise function to define eps and gamma.

        The function updates the `self.eps` and `self.g` attributes based on the selected structure.
        """

        self.g0 = g0  
        self.L0 = L0  
        self.struct = wind_struct

        if callable(self.struct):  # Custom function
            self.L = eps_grid(self.L0, self.theta, self.phi, struct=self.struct)
            self.g = gamma_grid(self.g0, self.theta, self.phi, struct=self.struct)

        elif self.struct in (1, 'tophat'):  # Tophat
            self.L = eps_grid(self.L0, self.theta, self.phi, struct='tophat', cutoff=self.theta_cut)
            self.g = gamma_grid(self.g0, self.theta, self.phi, struct='tophat', cutoff=self.theta_cut)

        elif self.struct in (2, 'gaussian'):  # Gaussian
            self.L = eps_grid(self.L0, self.theta, self.phi, theta_c=self.theta_c, struct='gaussian', cutoff=self.theta_cut)
            self.g = gamma_grid(self.g0, self.theta, self.phi, theta_c=self.theta_c, struct='gaussian', cutoff=self.theta_cut)

        elif self.struct in (3, 'powerlaw'):  # Power-law
            l = 2
            self.L = eps_grid(self.L0, self.theta, self.phi, theta_c=self.theta_c, k=l, struct='powerlaw', cutoff=self.theta_cut)
            self.g = gamma_grid(self.g0, self.theta, self.phi, theta_c=self.theta_c, k=l, struct='powerlaw', cutoff=self.theta_cut)

        else: 
            raise ValueError(f"Unsupported jet structure type: {self.struct}. Use 'tophat', 'gaussian', 'powerlaw', or a custom function.")

    def normalize(self, L0):
        """
        Normalizes the wind luminosity profile to match the given luminosity.

        This method scales the jet's luminosity profile (per solid angle) such that the observed on-axis 
        luminosity equals a given value.

        Parameters:
        L (float): The target luminosity to normalize to.
        """

        self.observer()

        # Compute the normalization factor
        A = L0 / self.L_prime_los
    
        # Apply the normalization
        self.L *= A
        
        print('Normalized L0:', self.L[0][0])


    def observer(self, theta_los=0, phi_los=0):
        """
        Compute observer-frame wind emission, including luminosity and spectra.

        Parameters:
            theta_los (float): Observer polar angle (line of sight).
            phi_los (float): Observer azimuthal angle.
        """

        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)

        # Compute Doppler factors
        dopp_on = doppf(self.g, 0)  # On-axis Doppler factor
        dopp_off = doppf(
            self.g,
            angular_d(self.theta[los_coord[0], los_coord[1]], self.theta,
                    self.phi[los_coord[0], los_coord[1]], self.phi)
        )
        self.r_dopp = dopp_off / dopp_on

        # Doppler-boosted luminosity
        self.L_prime = self.L * self.r_dopp**2
        self.L_prime4 = self.L * self.r_dopp**4
        L_obs_per_sa = self.L * self.r_dopp**4

        mask_cut = (self.theta < self.theta_cut) | (self.theta > np.pi - self.theta_cut)

        # self.L_prime_los = np.sum(L_cut * dOmega_cut * r_dopp_cut**0)
        # self.L_prime_los_full = np.sum(self.L_prime * self.dOmega * self.r_dopp**0) 

        self.L_prime_los = np.sum(L_obs_per_sa[mask_cut] * self.dOmega[mask_cut])
        self.L_prime_los_full = np.sum(L_obs_per_sa * self.dOmega)

        weight = np.sum(self.dOmega )

        self.L_prime_los /= weight
        self.L_prime_los_full /= weight

        # print('Total L_eff in los:', self.L_prime_los)

        tau = np.exp(-10 * self.engine.tau)  # Smoothly transitions from 0 to 1

        self.L_obs = self.L_prime_los_full * tau * (self.engine.Omega / self.engine.Omega_0)**4

        # Total observed luminosity
        self.L_X_tot = np.where(
            self.engine.t > self.engine.t_coll,
            0,
            (1 - tau) * self.L_prime_los * (self.engine.Omega / self.engine.Omega_0)**4 +
            tau * self.L_prime_los_full * (self.engine.Omega / self.engine.Omega_0)**4
        )