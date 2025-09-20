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
    def __init__(self, n_theta=100, n_phi=100, g0=200, E_iso=1e53, eps0=1e53, theta_c=np.pi/2, k=0, theta_cut=np.pi/2, jet_struct=0):
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
        theta_bounds = [0, np.pi]
        phi_bounds = [0, 2 * np.pi]
        
        # Generate a grid for theta and phi, based on the specified grid points
        self.theta_grid, self.phi_grid = coord_grid(n_theta, n_phi, theta_bounds, phi_bounds)
        
        # Compute cell-centered theta and phi values
        self.theta = 0.25 * (self.theta_grid[:-1, :-1] + self.theta_grid[1:, :-1] +
                            self.theta_grid[:-1, 1:] + self.theta_grid[1:, 1:])
        self.phi   = 0.25 * (self.phi_grid[:-1, :-1] + self.phi_grid[1:, :-1] +
                            self.phi_grid[:-1, 1:] + self.phi_grid[1:, 1:])

        # Compute the differential solid angle (dOmega) for each grid cell
        dtheta = np.gradient(self.theta, axis=0)
        dphi = np.gradient(self.phi, axis=1)
        self.dOmega = np.sin(self.theta) * dtheta * dphi

        # Set the core and cutoff angle for the jet
        self.theta_c = theta_c
        self.theta_cut = theta_cut
        # only used for PL profile
        self.k = k

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

        struct_map = {1: 'tophat', 2: 'gaussian', 3: 'powerlaw'}
        if callable(self.struct):
            struct = self.struct
        elif self.struct in struct_map:
            struct = struct_map[self.struct]
        elif isinstance(self.struct, str) and self.struct.lower() in struct_map.values():
            struct = self.struct.lower()
        else:
            raise ValueError(f"Invalid structure type: {self.struct}")

        cutoff = self.theta_cut

        self.eps = eps_grid(self.eps0, self.theta, self.phi, theta_c=self.theta_c, k=self.k, struct=struct, cutoff=cutoff)
        E_iso_profile = eps_grid(self.E_iso, self.theta, self.phi, theta_c=self.theta_c, k=self.k, struct=struct, cutoff=cutoff)
        # self.g = gamma_grid(self.g0, self.theta, k=k, struct=struct, cutoff=cutoff)

        # self.g = gamma_grid(self.g0 , self.theta, k=0, struct='powerlaw')
        self.g = lg11(E_iso_profile)

    def normalize(self, E_iso):
        """
        Normalizes the jet energy profile to match the given isotropic-equivalent energy.

        This method scales the jet's energy profile (per solid angle) such that the observed on-axis 
        isotropic-equivalent energy equals a given value.

        Parameters:
        E_iso (float): The target isotropic-equivalent energy to normalize to.
        """

        # Compute the isotropic equivalent energy per solid angle
        e_iso_grid_N = e_iso_grid(self.theta, self.phi, 0, self.g, self.eps, self.theta_cut, self.dOmega)
        e_iso_grid_S = e_iso_grid(self.theta, self.phi, np.pi, self.g, self.eps, self.theta_cut, self.dOmega)
        self.e_iso_grid = e_iso_grid_N + e_iso_grid_S
        # Compute the normalization factor
        A = E_iso / self.e_iso_grid[0]

        # Apply the normalization
        self.eps *= A
        
        # Recalculate E_iso per grid
        e_iso_grid_N = e_iso_grid(self.theta, self.phi, 0, self.g, self.eps, self.theta_cut, self.dOmega)
        e_iso_grid_S = e_iso_grid(self.theta, self.phi, np.pi, self.g, self.eps, self.theta_cut, self.dOmega)
        self.e_iso_grid = e_iso_grid_N + e_iso_grid_S

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
        D_off = doppf(self.g, angular_d(self.theta[los_coord[0], los_coord[1]], self.theta, self.phi[los_coord[0], los_coord[1]], self.phi))
        self.R_D = D_off / D_on

        # EATS
        beta = gamma2beta(self.g)
        theta_obs = angular_d(theta_los, self.theta, phi_los, self.phi)
        t_lab = self.t[np.newaxis, np.newaxis, :]
        R_em = c * t_lab * beta[..., np.newaxis] / (1 - beta[..., np.newaxis])
        self.t_obs = self.t + R_em / c * (1 - np.cos(theta_obs)[..., np.newaxis])
        
        EN_E_per_sa_obs = self.EN_E * self.R_D[..., np.newaxis]**3
        L_gamma_per_sa_obs = self.L_gamma * self.R_D[..., np.newaxis]**4
        L_X_per_sa_obs = self.L_X * self.R_D[..., np.newaxis]**4

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
        print('E_peak:', self.E[np.argmax(self.E * self.spec_tot)], 'eV')
        # print('L_gamma_peak:', np.max(self.L_gamma_tot))
        # print('t_peak', self.t[np.argmax(self.L_gamma_tot)])

        self.S_prime = self.S_gamma * self.R_D
        self.S_prime3 = self.S_gamma * self.R_D**3

    def refine_grid(self, theta_los, phi_los):
        """
        Rotate the grid so that the Doppler-brightest spot aligns with the new pole.
        """
        # --- Step 1: Compute R_D on cell centers ---
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)
        theta_los_val = self.theta[los_coord[0], los_coord[1]]
        phi_los_val   = self.phi[los_coord[0], los_coord[1]]

        D_on = doppf(self.g, 0)
        D_off = doppf(self.g, angular_d(theta_los_val, self.theta, phi_los_val, self.phi))
        R_D = D_off / D_on

        # --- Step 2: Find max R_D location ---
        imax, jmax = np.unravel_index(np.argmax(R_D), R_D.shape)
        theta_peak = self.theta[imax, jmax]
        phi_peak = self.phi[imax, jmax]

        # --- Step 3: Rotate edges (theta_grid, phi_grid) ---
        x = np.sin(self.theta_grid) * np.cos(self.phi_grid)
        y = np.sin(self.theta_grid) * np.sin(self.phi_grid)
        z = np.cos(self.theta_grid)
        xyz = np.stack([x, y, z], axis=-1)

        target_vec = np.array([
            np.sin(theta_peak) * np.cos(phi_peak),
            np.sin(theta_peak) * np.sin(phi_peak),
            np.cos(theta_peak)
        ])
        north_pole = np.array([0.0, 0.0, 1.0])

        rot_axis = np.cross(north_pole, target_vec)
        axis_norm = np.linalg.norm(rot_axis)
        if axis_norm < 1e-12:
            xyz_rot = xyz
        else:
            rot_axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(north_pole, target_vec), -1.0, 1.0))
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]],
                        [-rot_axis[1], rot_axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            xyz_rot = xyz @ R.T

        # --- Step 4: Convert back to spherical ---
        x_rot, y_rot, z_rot = xyz_rot[...,0], xyz_rot[...,1], xyz_rot[...,2]
        self.theta_grid = np.arccos(np.clip(z_rot, -1.0, 1.0))
        self.phi_grid = np.arctan2(y_rot, x_rot) % (2*np.pi)

        # --- Step 5: Compute cell-centered theta/phi correctly ---
        xyz_cells = 0.25 * (xyz_rot[:-1, :-1] + xyz_rot[1:, :-1] +
                            xyz_rot[:-1, 1:] + xyz_rot[1:, 1:])
        x_c, y_c, z_c = xyz_cells[...,0], xyz_cells[...,1], xyz_cells[...,2]
        self.theta = np.arccos(np.clip(z_c, -1.0, 1.0))
        self.phi = np.arctan2(y_c, x_c) % (2*np.pi)

        # --- Step 6: Compute dOmega from rotated edges ---
        dtheta = self.theta_grid[1:, :-1] - self.theta_grid[:-1, :-1]
        dphi   = self.phi_grid[:-1, 1:] - self.phi_grid[:-1, :-1]
        self.dOmega = np.sin(0.25*(self.theta_grid[:-1, :-1] + self.theta_grid[1:, 1:])) * dtheta * dphi