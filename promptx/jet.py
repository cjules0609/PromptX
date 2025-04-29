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

    Parameters:
        n_theta (int): Number of theta grid points.
        n_phi (int): Number of phi grid points.
        g0 (float): Initial Lorentz factor on the jet axis.
        E_iso (float): Isotropic equivalent energy for Liang-Ghirlanda relation.
        eps0 (float): Peak energy per solid angle on the jet axis.
        theta_c (float): Core angle for structured jets.
        theta_cut (float): Jet cutoff angle.
        struct (int): Structure type (0: tophat, 1: gaussian, 2: powerlaw).
    """
    def __init__(self, n_theta=100, n_phi=100, g0=200, E_iso=1e53, eps0=1e53, theta_c=np.pi/2, theta_cut=np.pi/2, struct=0):
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

        # Depending on the selected jet structure, define the energy per solid angle and Lorentz factor
        self.eps0 = eps0
        self.g0 = g0
        if struct == 0:  # Tophat
            self.eps = eps_grid(self.eps0, 0, self.theta, struct='pl', cutoff=theta_cut)  # Uniform energy per solid angle profile
            self.g = gamma_grid(g0, 0, self.theta, struct='pl', cutoff=theta_cut)  # Uniform Lorentz factor profile

        elif struct == 1:  # Gaussian
            sigma = theta_c # Gaussian width
            self.eps = eps_grid(self.eps0, sigma, self.theta, struct='gaussian', cutoff=theta_cut)  # Gaussian energy per solid angle profile
            E_iso_profile = eps_grid(E_iso, sigma, self.theta, struct='gaussian', cutoff=theta_cut)  # Normalized energy per solid angle profile
            self.g = lg11(E_iso_profile)  # Lorentz factor profile based on Liang-Ghirlanda 2011

        elif struct == 2:  # Power-law
            l = 10  # Power-law index
            self.eps = eps_grid(self.eps0, l, self.theta, struct='pl', cutoff=theta_cut)  # Power-law energy per solid angle profile
            E_iso_profile = eps_grid(E_iso, l, self.theta, struct='gaussian', cutoff=theta_cut)  # Normalized energy per solid angle profile
            self.g = lg11(E_iso_profile)  # Lorentz factor profile based on Liang-Ghirlanda 2011

        self.e_iso_grid = e_iso_grid(self.theta, self.phi, self.g, self.eps, self.dOmega)
        self.normalize(E_iso)
        self.create_obs_grid(amati_index=0.4)
    
    def normalize(self, E_iso):
        """
        Normalizes the jet energy profile to match the given isotropic-equivalent energy.

        This method scales the jet's energy profile (per solid angle) such that the observed on-axis 
        isotropic-equivalent energy equals a given value.

        Parameters:
        E_iso (float): The target isotropic-equivalent energy to normalize to.
        """
 
        # Compute the normalization factor
        A = E_iso / self.e_iso_grid[0]

        # Apply the normalization
        self.eps *= A
        
        # Recalculate E_iso per grid
        self.e_iso_grid = e_iso_grid(self.theta, self.phi, self.g, self.eps, self.dOmega)
        
        # Print the normalized value of eps0 for verification
        print('Normalized eps0:', self.eps[0][0])

    def create_obs_grid(self, amati_index=0.5):
        """
        Generate observer-frame spectral and temporal grids for gamma-ray and X-ray bands.

        This method computes the on-grid energy array, photon number spectrum, time array,
        luminosity light curves, and integrated fluences for two energy ranges:
        - Gamma-rays: 10 keV to 1000 keV
        - X-rays: 0.3 keV to 10 keV

        The Amati relation is used to set the spectral peak energy based on the input `amati_index`.

        Args:
            amati_index (float, optional): Slope of the Amati relation to use when determining
                the rest-frame peak energy. Default is 0.5.
        """

        # Calculate on-grid spectrum and light curve for gamma rays (10e3 - 1000e3 eV)
        self.E, self.N_E, self.t, self.L_gamma, self.S_gamma = obs_grid(self.eps, self.e_iso_grid, amati_index, e_1=10e3, e_2=1000e3)

        # Calculate on-grid spectrum and light curve for X-rays (0.3e3 - 10e3 eV)
        _, _, _, self.L_X, self.S_X = obs_grid(self.eps, self.e_iso_grid, amati_index, e_1=0.3e3, e_2=10e3)

    def observer(self, theta_los=0, phi_los=0):
        """
        Calculates observer-frame properties of the jet at a given line of sight.

        Parameters:
            theta_los (float): Line-of-sight polar angle (in radians). Default is 0.
            phi_los (float): Line-of-sight azimuthal angle (in radians). Default is 0.
        """

        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)

        # Compute ratio of Doppler factors
        D_on = doppf(self.g, 0)
        D_off = doppf(self.g, angular_d(self.theta[los_coord[1], los_coord[0]], self.theta, self.phi[los_coord[1], los_coord[0]], self.phi))
        R_D = D_off / D_on

        # Adjust time according to geometric time delay
        theta_obs = angular_d(theta_los, self.theta, phi_los, self.phi)
        beta = gamma2beta(self.g)
        t_em = self.t[np.newaxis, np.newaxis, :]  # shape: (1, 1, n_time)   
        R_IS = c * t_em / (1 - beta[..., np.newaxis])
        dt_geo = R_IS / beta[..., np.newaxis] / c * (1 - beta[..., np.newaxis] * np.cos(theta_obs)[..., np.newaxis])
        dt_geo = np.nan_to_num(dt_geo)
        self.t_obs = (self.t + dt_geo)

        # Adjust spectrum and LC by Doppler ratio
        self.N_E_obs = self.N_E * R_D[..., np.newaxis]**3 * self.dOmega[..., np.newaxis]
        self.L_gamma_obs = self.L_gamma * R_D[..., np.newaxis]**3 * self.dOmega[..., np.newaxis]
        self.L_X_obs = self.L_X * R_D[..., np.newaxis]**3 * self.dOmega[..., np.newaxis]
        
        # Sum the spectra over emitting regions
        self.spec_tot = np.sum(np.where((self.eps[..., np.newaxis] > 0), self.N_E_obs, 0), axis=(0, 1))

        # Interpolate the light curves for gamma-ray and X-ray emissions
        self.t, self.L_gamma_tot = interp_lc(self.t_obs, self.L_gamma_obs)
        _, self.L_X_tot = interp_lc(self.t_obs, self.L_X_obs)

        # Weight by solid angle
        weight = np.sum(np.where((self.eps[..., np.newaxis] > 0), R_D[..., np.newaxis]**2 * self.dOmega[..., np.newaxis], 0), axis=(0, 1))
        self.L_gamma_tot /= weight
        self.L_X_tot /= weight
        self.spec_tot /= weight

        # Calculate observed energy per solid angle
        self.eps_bar_gamma = np.sum(np.where((self.eps > 0), int_spec(self.E, self.spec_tot, E_min=10e3, E_max=1e6) * R_D**3 * self.dOmega, 0)) / np.sum(np.where((self.eps > 0), R_D**2 * self.dOmega, 0))

        # Calculate isotropic-equivalent properties
        self.L_gamma_tot *= 4 * np.pi
        self.L_X_tot *= 4 * np.pi
        self.E_iso_obs = 4 * np.pi * self.eps_bar_gamma
        # print('Observed E_iso:', self.E_iso_obs)

        self.S_prime = self.S_gamma * R_D
        self.S_prime3 = self.S_gamma * R_D**3