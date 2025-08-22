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

from numpy import int_, newaxis
from .helper import *
from .const import *

class Jet:
    """
    Represents a relativistic jet launched by a central engine, characterized
    by its energy and Lorentz factor structure as a function of polar angle.
    """
    def __init__(self, n_theta=100, n_phi=100, g0=200, E_iso=1e53, eps0=1e53, theta_c=np.pi/2, theta_cut=np.pi/2, jet_struct=0):
        """
        Initializes the jet model by setting up the grid, defining energy and Lorentz factor profiles,
        and normalizing the energy distribution.

        Parameters:
            n_theta (int): Number of polar (theta) grid points.
            n_phi (int): Number of azimuthal (phi) grid points.
            g0 (float): Lorentz factor normalization.
            E_iso (float): On-axis isotropic equivalent energy for normalization.
            eps0 (float): On-axis energy per solid angle.
            theta_c (float): Core angle for the jet.
            theta_cut (float): Cutoff angle for the jet structure.
            jet_struct (str or function): Structure type, either 'gaussian', 'powerlaw', or a custom function.
            
        Initializes the following attributes:
            theta_grid (ndarray): 2D grid of theta values for the jet.
            phi_grid (ndarray): 2D grid of phi values for the jet.
            theta (ndarray): 1D array of cell-centered theta values.
            phi (ndarray): 1D array of cell-centered phi values.
            dOmega (ndarray): Differential solid angle for each grid cell.
            theta_c (float): Core angle of the jet.
            theta_cut (float): Cutoff angle for the wind structure.
            eps (ndarray): Energy per solid angle profile of the jet.
            g (ndarray): Lorentz factor profile of the jet.
        """

        # Define the bounds for theta (polar angle) and phi (azimuthal angle)
        theta_bounds = [0, np.pi/2]
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

        # Set the core angle (theta_c) and cutoff angle (theta_cut) for the jet
        self.theta_c = theta_c
        self.theta_cut = theta_cut

        # Define Lorentz factor and energy per solid angle profiles
        self.define_structure(g0, eps0, E_iso, jet_struct)

        # Normalize
        self.normalize(self.E_iso)

    def define_structure(self, g0, eps0, E_iso, jet_struct):
        """
        Defines the structure of the wind's energy and Lorentz factor profiles based on the specified profile type.

        Parameters:
        g0 (float): The initial Lorentz factor normalization.
        eps0 (float): The initial energy per solid angle (before applying structure).
        E_iso (float): The isotropic-equivalent energy used for the Gaussian and power-law structures.
        struct (int or function): The structure type, where:
            - 1: Tophat
            - 2: Gaussian
            - 3: Power-law
            - function: A custom piecewise function to define eps and gamma.

        The function updates the `self.eps` and `self.g` attributes based on the selected structure.
        """

        self.g0 = g0  
        self.eps0 = eps0  
        self.E_iso = E_iso  
        self.struct = jet_struct

        if callable(self.struct):  # Check if struct is a function
            self.eps = eps_grid(self.eps0, self.theta, struct=self.struct)
            E_iso_profile = eps_grid(self.E_iso, self.theta, struct=self.struct)
            self.g = gamma_grid(self.g0, self.theta, struct=self.struct)

        elif self.struct == 1 or self.struct == 'tophat':  # Tophat
            self.eps = eps_grid(self.eps0, self.theta, k=0, struct='powerlaw')
            E_iso_profile = eps_grid(self.E_iso, self.theta, k=0, struct='powerlaw')
            self.g = gamma_grid(self.g0, self.theta, k=0, struct='powerlaw')

        elif self.struct == 2 or self.struct == 'gaussian':  # Gaussian
            sigma = self.theta_c
            self.eps = eps_grid(self.eps0, self.theta, k=sigma, struct='gaussian')
            E_iso_profile = eps_grid(self.E_iso, self.theta, k=sigma, struct='gaussian')
            self.g = gamma_grid(self.g0, self.theta, k=sigma, struct='gaussian')

        elif self.struct == 3 or self.struct == 'powerlaw':  # Power-law
            l = 2
            self.eps = eps_grid(self.eps0, self.theta, k=l, struct='powerlaw')
            E_iso_profile = eps_grid(self.E_iso, self.theta, k=l, struct='powerlaw')
            self.g = gamma_grid(self.g0, self.theta, k=l, struct='powerlaw')

        else: 
            raise ValueError(f"Unsupported jet structure type: {self.struct}. Use 'tophat', 'gaussian', 'powerlaw', or a custom function.")
        # self.g = gamma_grid(self.g0 , self.theta, k=0, struct='powerlaw')
        self.g = lg11 (E_iso_profile)

    def normalize(self, E_iso):
        """
        Normalizes the jet energy profile to match the given isotropic-equivalent energy.

        This method scales the jet's energy profile (per solid angle) such that the observed on-axis 
        isotropic-equivalent energy equals a given value.

        Parameters:
        E_iso (float): The target isotropic-equivalent energy to normalize to.
        """

        # Compute the isotropic equivalent energy per solid angle
        self.e_iso_grid = e_iso_grid(self.theta, self.phi, self.g, self.eps, self.theta_cut, self.dOmega)
        # Compute the normalization factor
        A = E_iso / self.e_iso_grid[0]

        # Apply the normalization
        self.eps *= A
        
        # Recalculate E_iso per grid
        self.e_iso_grid = e_iso_grid(self.theta, self.phi, self.g, self.eps, self.theta_cut, self.dOmega)

    def create_obs_grid(self, amati_a=0.41, amati_b=0.83):
        """
        Generate observer-frame spectral and temporal grids for gamma-ray and X-ray bands.

        This method computes the on-grid energy array, photon number spectrum, time array,
        luminosity light curves, and integrated fluences for two energy ranges:
        - Gamma-rays: 10 keV to 1000 keV
        - X-rays: 0.3 keV to 10 keV

        The Amati relation, E_p = 1e5 * 10**(amati_a * np.log10(e_iso_grid / 1e51) + amati_b)
        is used to set the spectral peak energy based on the input `amati_a` and `amati_b`.

        Args:
            amati_a (float, optional): Slope of the Amati relation to use when determining
                the rest-frame peak energy. 
            amati_b (float, optional): Intercept of the Amati relation to use when determining
                the rest-frame peak energy. 
        """

        # Calculate on-grid spectrum and light curve for gamma rays (10e3 - 1000e3 eV)
        self.E, self.EN_E, self.t, self.L_gamma, self.S_gamma = obs_grid(self.eps, self.e_iso_grid, amati_a=amati_a, amati_b=amati_b, e_1=10e3, e_2=1000e3)

        # Calculate on-grid spectrum and light curve for X-rays (0.3e3 - 10e3 eV)
        _, _, _, self.L_X, self.S_X = obs_grid(self.eps, self.e_iso_grid, amati_a=amati_a, amati_b=amati_b, e_1=0.3e3, e_2=10e3)

    def observer(self, theta_los=0, phi_los=0):
        """
        Calculates observer-frame properties of the jet at a given line of sight.

        Parameters:
            theta_los (float): Line-of-sight polar angle (in radians). 
            phi_los (float): Line-of-sight azimuthal angle (in radians). 
        """
        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)

        # Compute ratio of Doppler factors
        D_on = doppf(self.g, 0)
        D_off = doppf(self.g, angular_d(self.theta[los_coord[1], los_coord[0]], self.theta, self.phi[los_coord[1], los_coord[0]], self.phi))
        R_D = D_off / D_on

        # EATS
        beta = gamma2beta(self.g)
        theta_obs = angular_d(theta_los, self.theta, phi_los, self.phi)
        t_lab = self.t[np.newaxis, np.newaxis, :]
        R_em = c * t_lab * beta[..., np.newaxis] / (1 - beta[..., np.newaxis])
        self.t_obs = self.t + R_em / c * (1 - np.cos(theta_obs)[..., np.newaxis])
        
        EN_E_per_sa_obs = self.EN_E * R_D[..., np.newaxis]**3
        L_gamma_per_sa_obs = self.L_gamma * R_D[..., np.newaxis]**4
        L_X_per_sa_obs = self.L_X * R_D[..., np.newaxis]**4

        # Adjust spectrum and LC by Doppler ratio
        self.EN_E_obs = EN_E_per_sa_obs * self.dOmega[..., np.newaxis]
        self.L_gamma_obs = L_gamma_per_sa_obs * self.dOmega[..., np.newaxis]
        self.L_X_obs = L_X_per_sa_obs * self.dOmega[..., np.newaxis]

        # Sum the spectra over emitting regions
        self.spec_tot = 4 * np.pi * np.sum(self.EN_E_obs, axis=(0, 1))

        # Interpolate the light curves for gamma-ray and X-ray emissions
        self.t, self.L_gamma_tot = interp_lc(self.t_obs, self.L_gamma_obs)
        _, self.L_X_tot = interp_lc(self.t_obs, self.L_X_obs)

        # Weight by solid angle
        weight = np.sum(self.dOmega[..., np.newaxis], axis=(0, 1))
        self.L_gamma_tot /= weight
        self.L_X_tot /= weight
        self.spec_tot /= weight

        # Calculate observed energy per solid angle
        self.eps_bar_gamma = int_spec(self.E, self.spec_tot, E_min=10e3, E_max=1000e3)

        # Calculate isotropic-equivalent properties
        self.L_gamma_tot *= 4 * np.pi
        self.L_X_tot *= 4 * np.pi
        self.E_iso_obs = self.eps_bar_gamma
        self.L_iso_obs = int_lc(self.t, self.L_gamma_tot)

        print('Spectrum integral:', int_spec(self.E, self.spec_tot, E_min=10e3, E_max=1000e3))
        print('Light curve integral (gamma):', int_lc(self.t, self.L_gamma_tot))
        print('Light curve integral (X):', int_lc(self.t, self.L_X_tot))
        # print('L_gamma_peak:', np.max(self.L_gamma_tot))
        # print('t_peak', self.t[np.argmax(self.L_gamma_tot)])

        self.S_prime = self.S_gamma * R_D
        self.S_prime3 = self.S_gamma * R_D**3