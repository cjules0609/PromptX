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

class Jet:
    """
    Represents a relativistic jet launched by a central engine, characterized
    by its energy and Lorentz factor structure as a function of polar angle.
    """
    def __init__(self, n_theta=100, n_phi=100, g0=200, E_iso=1e53, eps0=1e53, theta_jet=np.pi/2, theta_cut=np.pi/2, jet_struct=0, **kwargs):
        """
        Initializes the jet model by setting up the grid, defining energy and Lorentz factor profiles,
        and normalizing the energy distribution.

        Parameters:
            n_theta (int): Number of polar (theta) grid points.
            n_phi (int): Number of azimuthal (phi) grid points.
            g0 (float): Lorentz factor normalization.
            E_iso (float): On-axis isotropic equivalent energy for normalization.
            eps0 (float): On-axis energy per solid angle.
            theta_jet (float): Core angle for the jet.
            theta_cut (float): Cutoff angle for the jet structure.
            jet_struct (str or function): Structure type, either 'gaussian', 'powerlaw', or a custom function.
            
        Initializes the following attributes:
            theta_grid (ndarray): 2D grid of theta values for the jet.
            phi_grid (ndarray): 2D grid of phi values for the jet.
            theta (ndarray): 1D array of cell-centered theta values.
            phi (ndarray): 1D array of cell-centered phi values.
            dOmega (ndarray): Differential solid angle for each grid cell.
            theta_jet (float): Core angle of the jet.
            theta_cut (float): Cutoff angle for the wind structure.
            eps (ndarray): Energy per solid angle profile of the jet.
            g (ndarray): Lorentz factor profile of the jet.
        """

        # Define the bounds for theta (polar angle) and phi (azimuthal angle)
        self.theta_bounds = [0, np.pi]
        self.phi_bounds = [0, 2 * np.pi]
        
        # Generate a grid for theta and phi, based on the specified grid points
        self.theta_grid, self.phi_grid = coord_grid(n_theta, n_phi, self.theta_bounds, self.phi_bounds)
        
        # Compute cell-centered theta and phi values
        self.theta = 0.25 * (self.theta_grid[:-1, :-1] + self.theta_grid[1:, :-1] +
                            self.theta_grid[:-1, 1:] + self.theta_grid[1:, 1:])
        self.phi   = 0.25 * (self.phi_grid[:-1, :-1] + self.phi_grid[1:, :-1] +
                            self.phi_grid[:-1, 1:] + self.phi_grid[1:, 1:])

        # Compute the differential solid angle (dOmega) for each grid cell
        theta_lo = self.theta_grid[:-1, :-1]
        theta_hi = self.theta_grid[1:, :-1]
        phi_lo   = self.phi_grid[:-1, :-1]
        phi_hi   = self.phi_grid[:-1, 1:]

        self.dOmega = (phi_hi - phi_lo) * (np.cos(theta_lo) - np.cos(theta_hi))

        # Set the core and cutoff angle for the jet
        self.theta_jet = theta_jet
        self.theta_cut = theta_cut

        self.E_iso = E_iso

        # Define Lorentz factor and energy per solid angle profiles
        self.define_structure(g0, eps0, jet_struct, **kwargs)

        # Normalize
        self.normalize(self.E_iso)

    def define_structure(self, g0, eps0, jet_struct, **kwargs):
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
        self.struct = jet_struct

        # --- Compute north and south contributions ---
        eps_north = eps_grid(
            self.eps0, self.theta, self.phi,
            theta_jet=self.theta_jet,
            struct=self.struct,
            cutoff=self.theta_cut,
            **kwargs
        )

        eps_south = eps_grid(
            self.eps0, np.pi - self.theta, self.phi,
            theta_jet=self.theta_jet,
            struct=self.struct,
            cutoff=self.theta_cut,
            **kwargs
        )

        # Combine both jets
        self.eps = eps_north + eps_south

        # Lorentz factor profile based on E_iso profile
        self.g = lg11(self.eps, self.theta, self.theta_cut)

    def normalize(self, E_iso):
        """
        Normalizes the jet energy profile to match the given isotropic-equivalent energy.

        This method scales the jet's energy profile (per solid angle) such that the observed on-axis 
        isotropic-equivalent energy equals a given value.

        Parameters:
        E_iso (float): The target isotropic-equivalent energy to normalize to.
        """
        # Initial guess (2D)
        g = self.g.copy()
        eps = self.eps.copy()

        # Initial E_iso calculation and normalization (2D)
        e_iso_grid = calc_e_iso_grid(self.theta, self.phi, g, eps, self.theta_cut, self.dOmega)
        A = E_iso / e_iso_grid[0, 0]
        eps *= A

        # Iterative loop
        tol = 1e-2
        max_iter = 10
        alpha = 0.5

        for i in range(max_iter):
            g_old = g.copy()

            # 1. Compute E_iso grid (2D)
            e_iso_grid = calc_e_iso_grid(self.theta, self.phi, g, eps, self.theta_cut, self.dOmega)

            # 2. Update Gamma with under-relaxation
            g_new = lg11(e_iso_grid, self.theta, self.theta_cut)
            g = (1 - alpha) * g_old + alpha * g_new

            # 3. Rescale eps to match target E_iso (use first element)
            A = E_iso / e_iso_grid[0, 0]
            eps *= A

            # 4. Compute max relative change in Gamma
            rel_diff = np.max(np.abs(g - g_old)/g_old)
            # print(f"Iteration {i+1}: max relative difference = {rel_diff:.3e}")

            if rel_diff < tol:
                # print(f"Converged after {i+1} iterations")
                break
        else:
            print("Warning: did not converge")

        # Save results
        self.g = g
        self.beta = gamma2beta(self.g)
        self.D_on = doppf(self.g, 0)
        self.eps = eps
        self.e_iso_grid = e_iso_grid
        self.E_iso = e_iso_grid[0, 0]

    def create_obs_grid(self, amati_a=0.41, amati_b=0.83, tau_1=0.1, tau_2=0.35, e_1=10e3, e_2=1000e3):
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

        # Calculate on-grid spectrum and light curve for gamma rays (1e3 - 10e6 eV)
        self.E, self.EN_E, self.t, self.L_gamma, self.S_gamma = obs_grid(self.eps, self.e_iso_grid, amati_a=amati_a, amati_b=amati_b, e_1=1e3, e_2=10e6, tau_1=tau_1, tau_2=tau_2)

        # Calculate on-grid spectrum and light curve for X-rays (0.3e3 - 10e3 eV)
        self.E_X, self.EN_E_X, self.t_X, self.L_X, self.S_X = obs_grid(self.eps, self.e_iso_grid, amati_a=amati_a, amati_b=amati_b, e_1=e_1, e_2=e_2, tau_1=tau_1, tau_2=tau_2)
                
    def observer(self, theta_los=0, phi_los=0):
        """
        Calculates observer-frame properties of the jet at a given line of sight.

        Parameters:
            theta_los (float): Line-of-sight polar angle (in radians). 
            phi_los (float): Line-of-sight azimuthal angle (in radians). 
        """
        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)
        theta_obs = angular_d(self.theta[los_coord[0], 0], self.theta, self.phi[0, los_coord[1]], self.phi)
        D_off = doppf(self.g, theta_obs)
        self.R_D = D_off / self.D_on

        t_eng = self.t_X
        self.t_obs = t_eng * (1 - self.beta[..., np.newaxis] * np.cos(theta_obs)[..., np.newaxis]) / (1 - self.beta[..., np.newaxis])
        
        EN_E_per_sa_obs = self.EN_E_X * self.R_D[..., np.newaxis]**3
        L_gamma_per_sa_obs = self.L_gamma * self.R_D[..., np.newaxis]**4
        L_X_per_sa_obs = self.L_X * self.R_D[..., np.newaxis]**4

        # Adjust spectrum and LC by Doppler ratio
        self.EN_E_obs = EN_E_per_sa_obs * self.dOmega[..., np.newaxis]
        self.L_gamma_obs = L_gamma_per_sa_obs * self.dOmega[..., np.newaxis]
        self.L_X_obs = L_X_per_sa_obs * self.dOmega[..., np.newaxis]

        # Sum the spectra over emitting regions
        self.spec_tot = np.sum(self.EN_E_obs, axis=(0, 1))

        # Interpolate the light curves for gamma-ray and X-ray emissions
        self.t, self.L_gamma_tot = interp_lc(self.t_obs, self.L_gamma_obs)
        self.t_X, self.L_X_tot = interp_lc(self.t_obs, self.L_X_obs)

        # Weight by solid angle
        weight = np.sum(self.dOmega[..., np.newaxis], axis=(0, 1))
        self.L_gamma_tot /= weight
        self.L_X_tot /= weight
        self.spec_tot /= weight

        # Calculate observed energy per solid angle
        self.eps_bar_gamma = int_spec(self.E, self.spec_tot, E_min=1e3, E_max=10e6)

        # Calculate isotropic-equivalent properties
        self.E_iso_obs = 4 * np.pi * self.eps_bar_gamma
        self.L_gamma_tot *= 4 * np.pi
        self.L_X_tot *= 4 * np.pi
        self.L_iso_obs = int_lc(self.t, self.L_gamma_tot)

        print('Spectrum integral (E_iso_obs):', self.E_iso_obs)
        print('Light curve integral (L_iso_obs):', self.L_iso_obs)
        # print('Light curve integral (X):', int_lc(self.t, self.L_X_tot))
        print('E_peak:', self.E[np.argmax(self.E * self.spec_tot)], 'eV')
        # print('L_gamma_peak:', np.max(self.L_gamma_tot))
        # print('t_peak', self.t[np.argmax(self.L_gamma_tot)])

        self.S_prime = self.S_gamma * self.R_D
        self.S_prime3 = self.S_gamma * self.R_D**3

    def refine_grid(self, theta_los, phi_los, n_theta=200, n_phi=100, rotate=False, resample=False):
        """
        Rotate the grid so that the Doppler-brightest spot aligns with the new pole.
        """
        
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)
        D_on = doppf(self.g, 0)
        theta_obs = angular_d(self.theta[los_coord[0], 0], self.theta, self.phi[0, los_coord[1]], self.phi)
        D_off = doppf(self.g, theta_obs)
        self.R_D = D_off / D_on
        if rotate:
            # --- Step 2: Find max R_D location ---
            imax, jmax = np.unravel_index(np.argmax(self.eps * self.R_D**3), self.R_D.shape)
            theta_peak = self.theta[imax, jmax]
            phi_peak = self.phi[imax, jmax]
            # print(f"Max R_D at (theta, phi) = ({theta_peak:.3f}, {phi_peak:.3f})")

            # --- Step 3: Rotate grid so that (theta_peak, phi_peak) -> (0, 0) ---
            theta_grid_rot, phi_grid_rot = rotate_spherical(self.theta_grid, self.phi_grid, theta_peak, phi_peak)
            theta_rot, phi_rot = rotate_spherical(self.theta, self.phi, theta_peak, phi_peak)

            # --- Step 4: Interpolate eps, g, e_iso_grid onto rotated grid ---
            # eps_rot = profile_interp(self.theta, self.phi, self.eps, theta_rot, phi_rot)
            # e_iso_rot = profile_interp(self.theta, self.phi, self.e_iso_grid, theta_rot, phi_rot)
            # g_rot = profile_interp(self.theta, self.phi, self.g, theta_rot, phi_rot)
            # R_D_rot = profile_interp(self.theta, self.phi, self.R_D, theta_rot, phi_rot)

            eps_rot = eps_grid(
                self.eps0, theta_rot, phi_rot,
                theta_jet=self.theta_jet,
                struct=self.struct,
                cutoff=self.theta_cut
            )

            e_iso_rot = eps_grid(
                self.E_iso, theta_rot, phi_rot,
                theta_jet=self.theta_jet,
                struct=self.struct,
                cutoff=self.theta_cut,
            )

            g_rot = lg11(e_iso_rot)
            e_iso_rot = calc_e_iso_grid(
                theta_rot, phi_rot, g_rot, eps_rot,
                theta_jet=self.theta_jet,
                dOmega=self.dOmega
            )

            R_D_rot = doppf(g_rot, theta_rot) / doppf(g_rot, 0)

            self.eps = eps_rot
            self.g = g_rot
            self.e_iso_grid = e_iso_rot
            self.R_D = R_D_rot
            
            # Update grid attributes with rotated grid
            # self.theta_grid = theta_grid_rot
            # self.phi_grid = phi_grid_rot
            # self.theta = theta_rot
            # self.phi = phi_rot

        if resample:
            n_theta_old, n_phi_old = self.theta.shape

            stride_theta = max(1, n_theta_old // n_theta)
            stride_phi   = max(1, n_phi_old // n_phi)

            # --- Downsample edges ---
            theta_grid_downsample = self.theta_grid[::stride_theta, ::stride_phi]
            phi_grid_downsample   = self.phi_grid[::stride_theta, ::stride_phi]

            # Ensure last row/col included
            if theta_grid_downsample.shape[0] != len(range(0, self.theta_grid.shape[0], stride_theta)):
                theta_grid_downsample = np.vstack([theta_grid_downsample, self.theta_grid[-1, ::stride_phi]])
            if phi_grid_downsample.shape[1] != len(range(0, self.phi_grid.shape[1], stride_phi)):
                phi_grid_downsample = np.hstack([phi_grid_downsample, self.phi_grid[::stride_theta, -1][:, None]])

            # --- Downsample cell centers (average of 4 corners) ---
            theta_downsample = 0.25 * (
                theta_grid_downsample[:-1, :-1] +
                theta_grid_downsample[1:, :-1] +
                theta_grid_downsample[:-1, 1:] +
                theta_grid_downsample[1:, 1:]
            )
            phi_downsample = 0.25 * (
                phi_grid_downsample[:-1, :-1] +
                phi_grid_downsample[1:, :-1] +
                phi_grid_downsample[:-1, 1:] +
                phi_grid_downsample[1:, 1:]
            )

            # --- Compute dOmega for each cell ---
            theta_lo = theta_grid_downsample[:-1, :-1]
            theta_hi = theta_grid_downsample[1:, :-1]
            phi_lo   = phi_grid_downsample[:-1, :-1]
            phi_hi   = phi_grid_downsample[:-1, 1:]

            dOmega_downsample = (phi_hi - phi_lo) * (np.cos(theta_lo) - np.cos(theta_hi))

            # --- Interpolate profiles onto new cell centers ---
            self.eps = profile_interp(self.theta, self.phi, self.eps, theta_downsample, phi_downsample, method='nearest')
            self.g = profile_interp(self.theta, self.phi, self.g, theta_downsample, phi_downsample, method='nearest')
            self.e_iso_grid = profile_interp(self.theta, self.phi, self.e_iso_grid, theta_downsample, phi_downsample, method='nearest')
            self.R_D = profile_interp(self.theta, self.phi, self.R_D, theta_downsample, phi_downsample, method='nearest')

            # --- Update grid attributes ---
            self.theta_grid = theta_grid_downsample
            self.phi_grid   = phi_grid_downsample
            self.theta = theta_downsample
            self.phi   = phi_downsample
            self.dOmega = dOmega_downsample