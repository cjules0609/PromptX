import numpy as np

from const import *
from helper import *
import os

def main(n_theta=500, n_phi=100, jet_E_iso=1e53, jet_eps0=1e53,
         theta_c=5, theta_cut=90, theta_los=0, phi_los=0,
         wind_norm=1, model_id=0):
    """
    Main simulation function to generate jet and wind models,
    calculate observer light curves, and optionally save output.

    Args:
        n_theta (int): Number of angular points in theta direction.
        n_phi (int): Number of angular points in phi direction.
        jet_E_iso (float): Isotropic-equivalent energy of the jet [erg].
        jet_eps0 (float): Total injected jet energy [erg].
        theta_c (float): Jet core angle [deg].
        theta_cut (float): Emission cutoff angle [deg].
        theta_los (float): Observer's line-of-sight theta [deg].
        phi_los (float): Observer's line-of-sight phi [deg].
        wind_norm (float): Wind normalization factor.
        model_id (int): Model identifier (0 = jet-only).

    Returns:
        tuple: (jet, wind) model instances.
    """
    print('---------------------JET---------------------')
    jet = Jet(
        g0=200,
        E_iso=jet_E_iso,
        eps0=jet_eps0,
        n_theta=n_theta,
        n_phi=n_phi,
        theta_c=theta_c,
        theta_cut=theta_cut,
        struct=1
    )
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    print('---------------------WIND---------------------')
    collapse = model_id != 1  # Collapse for all models except model_id == 1
    wind = Wind(
        g0=50,
        n_theta=n_theta * 10,
        n_phi=n_phi * 10,
        theta_cut=theta_cut,
        collapse=collapse
    )
    wind.observer(theta_los=theta_los, phi_los=phi_los, norm=wind_norm)

    if model_id:
        model_paths = {
            1: './out/bns1/',
            2: './out/bns2/',
            3: './out/bns3/',
            4: './out/bhns/'
        }
        path = model_paths.get(model_id, './out/')
        os.makedirs(path, exist_ok=True)

        save_data(jet, wind, theta_los, path=path, model_id=model_id)

    return jet, wind

    
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
        Omega = np.geomspace(self.Omega_0, 10, 1000)

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

class Jet:
    """
    Represents a relativistic jet launched by a central engine, characterized
    by its energy and Lorentz factor structure as a function of polar angle.

    Parameters:
        n_theta (int): Number of theta grid points.
        n_phi (int): Number of phi grid points.
        g0 (float): Initial Lorentz factor on the jet axis.
        E_iso (float): Isotropic equivalent energy for normalization.
        eps0 (float): Initial energy per solid angle on the jet axis.
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
            self.eps_bar = eps_grid(E_iso, sigma, self.theta, struct='gaussian', cutoff=theta_cut)  # Normalized energy per solid angle profile
            self.g = lg11(self.eps_bar)  # Lorentz factor profile based on Liang-Ghirlanda 2011
            # self.g = gamma_grid(self.g0, sigma, self.theta, struct='gaussian', cutoff=theta_cut)

        elif struct == 2:  # Power-law
            l = 10  # Power-law index
            self.eps = eps_grid(self.eps0, l, self.theta, struct='pl', cutoff=theta_cut)  # Power-law energy per solid angle profile
            self.eps_bar = eps_grid(E_iso, l, self.theta, struct='gaussian', cutoff=theta_cut)  # Normalized energy per solid angle profile
            self.g = lg11(self.eps_bar)  # Lorentz factor profile based on Liang-Ghirlanda 2011
            # self.g = gamma_grid(self.g0, l, self.theta, struct='pl', cutoff=theta_cut)

        # Compute intrinsic Doppler shift factors (on-axis and off-axis)
        D_on_intrinsic = doppf(self.g[0][0], 0)  # Doppler factor for on-axis observation
        D_off_intrinsic = doppf(self.g, angular_d(self.theta[0, 0], self.theta, self.phi[0, 0], self.phi))  # Doppler factor for off-axis observations

        # Compute ratio of off-axis Doppler factor to on-axis Doppler factor (for normalization)
        self.R_D_intrinsic = D_off_intrinsic / D_on_intrinsic
    
    def normalize(self, E_iso):
        """
        Normalizes the jet energy profile to match the given isotropic-equivalent energy.

        This method scales the jet's energy profile (per solid angle) such that the total energy 
        observed in the rest frame corresponds to the specified isotropic energy (E_iso).

        Parameters:
        E_iso (float): The target isotropic-equivalent energy to normalize to.
        """
        # Compute the total observed energy from the current energy per solid angle profile
        E_obs = np.sum(self.R_D_intrinsic**3 * self.eps * self.dOmega)

        # Compute the normalization factor (scaling the energy per solid angle to match E_iso)
        A = E_iso / E_obs

        # Apply the normalization to both the energy per solid angle and  initial value
        self.eps *= A
        self.eps0 *= A
        
        # Print the normalized value of eps0 for verification
        print('Normalized eps0:', self.eps0)

    def observer(self, theta_los=0, phi_los=0):
        """
        Calculates observer-frame properties of the jet at a given line of sight.

        Parameters:
            theta_los (float): Line-of-sight polar angle (in radians). Default is 0.
            phi_los (float): Line-of-sight azimuthal angle (in radians). Default is 0.
        """
        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)

        # Compute Doppler factor for on-axis observation
        D_on = doppf(self.g, 0)

        # Compute Doppler factor for off-axis observation at LoS
        D_off = doppf(self.g, angular_d(self.theta[los_coord[1], los_coord[0]], self.theta, self.phi[los_coord[1], los_coord[0]], self.phi))

        # Compute ratio of Doppler factors
        R_D = D_off / D_on

        # Calculate on-grid spectrum and light curve for gamma rays (10e3 - 1000e3 eV)
        self.E, self.N_E, self.t_obs, L_gamma, S = obs_grid(self.eps, R_D, self.R_D_intrinsic, e_1=10e3, e_2=1000e3)

        # Calculate on-grid spectrum and light curve for X-rays (0.3e3 - 10e3 eV)
        _, _, _, L_X, _ = obs_grid(self.eps, R_D, self.R_D_intrinsic, e_1=0.3e3, e_2=10e3)

        # Adjust spectrum by Doppler ratio and sum over the emitting regions
        self.N_E *= R_D[..., np.newaxis]**3 * self.dOmega[..., np.newaxis]
        self.spec_tot = np.sum(np.where(np.isfinite(self.N_E) & (self.eps[..., np.newaxis] > 0), self.N_E, 0), axis=(0, 1))

        # Adjust gamma-ray and X-ray luminosities by Doppler ratio
        self.L_gamma_obs = L_gamma * R_D[..., np.newaxis]**3 * self.dOmega[..., np.newaxis]
        self.L_X_obs = L_X * R_D[..., np.newaxis]**3 * self.dOmega[..., np.newaxis]

        # Interpolate the light curves for gamma-ray and X-ray emissions
        self.t, self.L_gamma_tot = interp_lc(self.t_obs, self.L_gamma_obs)
        _, self.L_X_tot = interp_lc(self.t_obs, self.L_X_obs)

        # Sum the observed energy of emitting regions
        self.S_obs = np.sum(np.where(np.isfinite(S) & (self.eps > 0), S * R_D**3 * self.dOmega, 0))
        # print('Observed total energy:', self.S_obs)

class Wind(Magnetar):
    """
    Represents a magnetar-powered relativistic wind, with energy and Lorentz factor
    profiles structured as a function of polar angle.

    Inherits from:
        Magnetar: Provides magnetar-related properties like magnetic field, spin, etc.

    Parameters:
        n_theta (int): Number of polar (theta) grid points.
        n_phi (int): Number of azimuthal (phi) grid points.
        g0 (float): Lorentz factor normalization.
        eps0 (float): Initial energy per solid angle (overwritten by magnetar spin-down formula).
        theta_cut (float): Angular cutoff for the wind structure.
        collapse (bool): If True, the magnetar collapses at some time, modifying evolution.
    """
    def __init__(self, n_theta=100, n_phi=100, g0=50, eps0=1e49, theta_cut=np.pi/2, collapse=False):
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
        eta = 0.01   # Conversion efficiency
        self.eps0 = eta * (self.engine.B_p**2 * self.engine.R**6 * self.engine.Omega_0**4) / (6 * c**3)

        # Assign isotropic energy profile
        self.eps = eps_grid(self.eps0, 0, self.theta, struct='pl', cutoff=theta_cut)

        # Assign isotropic Lorentz factor profile 
        self.g0 = g0
        self.g = self.g0 * gamma_grid(0, self.theta, struct='pl', cutoff=theta_cut)


    def observer(self, theta_los=0, phi_los=0, norm=1):
        """
        Compute observer-frame wind emission, including luminosity and spectra.

        Parameters:
            theta_los (float): Observer polar angle (line of sight).
            phi_los (float): Observer azimuthal angle.
            norm (float): Normalization factor for eps scaling.
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

        # Patch with maximum Doppler-boosted energy
        eps_prime_max_index = np.argmax(np.array(self.eps_prime))
        self.theta_max, self.phi_max = np.unravel_index(eps_prime_max_index, self.eps_prime.shape)

        # Compute effective line-of-sight energy of emitting regions (weighted average)
        eps_cut = self.eps_prime[self.eps_prime != 0]
        r_dopp_cut = self.r_dopp[self.eps_prime != 0]
        dOmega_cut = self.dOmega[self.eps_prime != 0]
        self.eps_prime_los = (np.sum(eps_cut * dOmega_cut * r_dopp_cut**2) / np.sum(dOmega_cut * r_dopp_cut**2))
        print('Total eps_eff in los:', self.eps_prime_los)


        # Compute base spin-down luminosity
        L_sd = 1e48 * (self.engine.Omega / self.engine.Omega_0) ** 4

        # On-axis wind emission
        self.L_on = np.where(self.engine.t > self.engine.t_coll, 0, L_sd)

        # Line-of-sight wind emission
        self.L_los = np.where(theta_los < self.theta_cut, L_sd, L_sd * np.exp(-10 * self.engine.tau))
        self.L_los = np.where(self.engine.t > self.engine.t_coll, 0, self.L_los)

        # Ratio of Doppler-boosted energy to on-axis energy
        self.r_eps = self.eps_prime_los / norm if self.eps_prime_los > 0 else np.nan

        # Doppler-boosted luminosity
        self.L_dopp = self.L_on * self.r_eps

        # Total observed luminosity
        self.L_X_tot = np.maximum(self.L_los, self.L_dopp)

        # Band function parameters
        E_p = 50e3
        E_min = 0.3e3
        E_max = 1e6
        alpha = -1.5
        beta = -3

        # On-axis spectrum
        self.E_on=np.geomspace(E_min, E_max, 200)
        self.N_E_on, self.A_on = spec(norm, E=self.E_on, E_p=E_p, E_det=np.geomspace(0.4e3, 5e3, 200), alpha=alpha, beta=beta) if theta_los > self.theta_cut else [0, 0]
        
        # LOS spectrum (same shape)
        self.E_los=np.geomspace(E_min, E_max, 200)
        self.N_E_los, self.A_los = spec(norm, E=self.E_los, E_p=E_p, E_det=np.geomspace(0.4e3, 5e3, 200), alpha=alpha, beta=beta) if theta_los > self.theta_cut else [0, 0]

        # Doppler-boosted spectrum (adjust E_p)
        self.E_dopp = np.geomspace(E_min, E_max, 200)
        self.N_E_dopp, self.A_dopp = spec(self.eps_prime_los, E=self.E_dopp, E_p=E_p * (self.r_eps)**0.5, E_det=np.geomspace(0.4e3, 5e3, 200), alpha=alpha, beta=beta)

def coord_grid(n_theta, n_phi, theta_bounds, phi_bounds):
    """
    Generates a grid of spherical coordinates (theta, phi).

    Args:
        n_theta (int): Number of theta (polar angle) grid points.
        n_phi (int): Number of phi (azimuthal angle) grid points.
        theta_bounds (list or tuple): Range of theta (polar angle) values [theta_min, theta_max] in radians.
        phi_bounds (list or tuple): Range of phi (azimuthal angle) values [phi_min, phi_max] in radians.

    Returns:
        tuple: theta (columns) and phi (rows) meshgrids.
    """
    # Generate theta values: cosine transformation for tighter spacing at N x pi for N = [0, 1, 2, ...]
    theta = -np.cos(np.linspace(theta_bounds[0], theta_bounds[1], n_theta)) * np.pi / 2 + np.pi / 2

    # Create meshgrid for spherical coordinates (theta, phi)
    theta, phi = np.meshgrid(theta, np.linspace(phi_bounds[0], phi_bounds[1], n_phi))
    return theta, phi


def gamma_grid(k, theta, struct='gaussian', cutoff=np.pi):
    """
    Generates grid of Lorentz factors based on given structure type and parameters.

    Args:
        k (float): Parameter that influences shape.
        theta (2D array): Meshgrid of theta (polar angle) values, typically generated by `coord_grid`.
        struct (str): Structure type, either 'gaussian' or 'pl' (power-law).
        cutoff (float): Cutoff angle beyond which Gamma = 1.

    Returns:
        2D array: Grid of Lorentz factors corresponding to provided structure.
    """
    # Select the appropriate structure type and calculate gamma values
    if struct == 'gaussian':
        g = gaussian(theta[:, :theta.shape[1]], k)
    elif struct == 'pl':
        g = powerlaw(theta[:, :theta.shape[1]], k)
    else:
        # Error handling for unsupported structure types
        print('struct must be gaussian or pl')
        return None

    # Set Gamma = 1 beyond cutoff
    g[np.abs(np.cos(theta)) < np.cos(cutoff)] = 1

    # Ensure Gamma >= 1
    g[g < 1] = 1

    return g


def eps_grid(eps0, k, theta, struct='gaussian', cutoff=np.pi):
    """
    Generates grid of energy per solid angle (eps) based on given structure type and parameters.

    Args:
        eps0 (float): Scaling factor for energy per solid angle.
        k (float): Parameter that influences shape.
        theta (2D array): Meshgrid of theta (polar angle) values, typically generated by `coord_grid`.
        struct (str): Structure type, either 'gaussian' or 'pl' (power-law).
        cutoff (float): Cutoff angle beyond which the eps = 0.

    Returns:
        2D array: Grid of energy per solid angle (eps) values corresponding to provided structure.
    """
    # Select appropriate structure type and calculate energy per solid angle
    if struct == 'gaussian':
        eps = gaussian(theta[:, :theta.shape[1]], k)
    elif struct == 'pl':
        eps = powerlaw(theta[:, :theta.shape[1]], k)
    else:
        # Error handling for unsupported structure types
        print('struct must be gaussian or pl')
        return None

    # Set eps = 0 for angles beyond cutoff
    eps[np.abs(np.cos(theta)) < np.cos(cutoff)] = 0

    return eps0 * eps

# on-grid observables
def obs_grid(eps, R_D, R_D_onaxis, e_1=0.3e3, e_2=10e3):
    """
    Computes the spectrum and light curve observed by an observer on a grid.
    
    Args:
        eps (2D array): Energy per solid angle (eps) grid values.
        R_D (2D array): Doppler factor grid values.
        R_D_onaxis (2D array): Doppler factor for the on-axis observer.
        e_1 (float): Minimum energy for the spectrum integration, default is 0.3 keV.
        e_2 (float): Maximum energy for the spectrum integration, default is 10 keV.

    Returns:
        tuple: Contains the following:
            - E (array): Energy grid used for the spectrum.
            - N_E_norm (array): Normalized Band spectrum.
            - t_obs (array): Time grid adjusted to the observer frame.
            - L_scaled (array): Scaled light curve.
            - S (array): Integrated spectrum over the detector energy band.
    """
    # SPECTRUM
    # Reference Band function parameters
    alpha, beta = -1, -2.3
    E_p_0 = 1e6 

    # Calculate the peak and cutoff energy based on the epsilon grid and normalization
    E_p = E_p_0 * (eps / eps[0][0]) ** 0.5   
    E_0 = E_p / (2 + alpha)

    # Define the energy grid for the spectrum integration
    E = np.geomspace(0.3e3, 1e6, 1000)

    # Compute the unnormalized Band spectrum
    N_E = band(E, alpha, beta, E_0)


    # Normalize spectrum to eps
    # This is the spectrum an on-grid observer would see
    eps_unit = int_spec(E, N_E, E_min=10e3, E_max=1e6)
    A_spec = eps / eps_unit
    N_E_norm = A_spec[..., np.newaxis] * N_E 

    # Integrate spectra of emitting regions over detector energy band
    S = int_spec(E, N_E_norm, E_min=e_1, E_max=e_2)
    S = np.nan_to_num(S, nan=0.0)

    # LIGHT CURVE
    # FRED function parameters
    a_1 = 0.2
    a_2 = 0.6

    # Generate FRED light curve
    t = np.geomspace(1e-3, 1e3, 1000)
    L = fred(t, a_1, a_2)

    # Adjust time to observer frame using the Doppler factor
    t_obs = t / R_D[..., np.newaxis]

    # Normalize fluence after
    S_unit = int_lc(t_obs, L)
    A_lc = S / S_unit
    L_scaled = A_lc[..., np.newaxis] * L 

    return E, N_E_norm, t_obs, L_scaled, S

if __name__ == '__main__':
    n_theta = 500
    n_phi = 100


    bns1 = {
        'theta_c'   :   np.deg2rad(5),
        'theta_cut'   :   np.deg2rad(35),
        'theta_los_list'   :   np.deg2rad([0, 30, 50]),
        'model_id'  :   1
    }
    bns2 = {
        'theta_c'   :   np.deg2rad(5),
        'theta_cut'   :   np.deg2rad(35),
        'theta_los_list'   :   np.deg2rad([0, 30, 50]),
        'model_id'  :   2
    }
    bns3 = {
        'theta_c'   :   np.deg2rad(5),
        'theta_cut'   :   np.deg2rad(35),
        'theta_los_list'   :   np.deg2rad([0, 30, 50]),
        'model_id'  :   3
    }
    bhns = {
        'theta_c'   :   np.deg2rad(15),
        'theta_cut'   :   np.deg2rad(80),
        'theta_los_list'   :   np.deg2rad([0, 30, 90]),
        'model_id'  :   4
    }
    
    E_gamma = 1e51 * (1 - np.cos(5 * np.pi/180))

    for model in [bns1, bns2, bns3, bhns]:
        theta_c = model['theta_c']
        theta_cut = model['theta_cut']
        model_id = model['model_id']
        theta_los_list = model['theta_los_list']
        phi_los = 0
    
        jet_E_iso = E_gamma / (1 - np.cos(theta_c))

        jet = Jet(E_iso=jet_E_iso, eps0=jet_E_iso, n_theta=n_theta, n_phi=n_phi, theta_c=theta_c, theta_cut=theta_cut, struct=1)
        jet.normalize(jet.eps0)
        jet.observer(theta_los=0, phi_los=0)
        jet_eps0 = jet.eps0
        
        wind = Wind(g0=50, n_theta=n_theta*10, n_phi=n_phi*10, theta_cut=theta_cut)
        wind.observer(theta_los=0, phi_los=0)
 
        for theta_los in theta_los_list:
            print('#----------------------{}------------------------#'.format(np.rad2deg(theta_los)))
            main(n_theta=n_theta, n_phi=n_phi, jet_E_iso=jet_E_iso, jet_eps0=jet_eps0, theta_c=theta_c, theta_cut=theta_cut, wind_norm=wind.eps_prime_los, theta_los=theta_los, phi_los=phi_los, model_id=model_id)