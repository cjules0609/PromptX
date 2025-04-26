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

import numpy as np
import csv

from const import *

def gamma2beta(gamma):
    return np.sqrt(1 - 1 / gamma**2)

def beta2gamma(beta):
    return 1 / np.sqrt(1 - beta**2)

def gaussian(x, sigma, mu=0):
    """
    Gaussian function.

    Args:
        x (array): Input values for Gaussian function.
        sigma (float): Standard deviation of Gaussian distribution.
        mu (float): Mean of Gaussian distribution, default is 0.

    Returns:
        array: Gaussian function evaluated at each point in x.
    """
    return np.exp(-((x - mu) ** 2 / (sigma ** 2)))

def powerlaw(x, k):
    """
    Power-law function for a given exponent k.

    Args:
        x (array): Input values for power-law function.
        k (float): Exponent of power-law.

    Returns:
        array: Power-law function evaluated at each point in x.
    """
    return x ** k

def doppf(g, theta):
    """
    Doppler factor for relativistic jets.

    Args:
        g (array): 2D array of Lorentz factor values at each theta x phi.
        theta (float): Angular distance to each theta x phi.

    Returns:
        array: 2D array of Doppler factors at each theta x phi.
    """
    # Calculate the Doppler factor based on the Lorentz factor (gamma) and angle (theta)
    return 1 / (g * (1 - np.sqrt(1 - 1 / g / g) * np.cos(theta)))

def band(E, alpha, beta, E_0):
    """
    Band function spectrum.

    Args:
        E (array): Energy values to compute the spectrum.
        alpha (float): Spectral index for low energies.
        beta (float): Spectral index for high energies.
        E_0 (float or 2D array): Cutoff energy - single or at each theta x phi.

    Returns:
        1D or 3D array: The Band function spectrum - single or at each theta x phi.
    """
    E = np.asarray(E)
    if np.isscalar(E_0) or E_0.ndim == 0:
        cond = E <= (alpha - beta) * E_0
        lowE = (E)**alpha * np.exp(-E / E_0)
        highE = ((alpha - beta) * E_0)**(alpha - beta) * np.exp(beta - alpha) * (E)**beta
    else:
        E_b = E[np.newaxis, np.newaxis, :]
        E_0_b = E_0[..., np.newaxis]

        cond = E_b <= (alpha - beta) * E_0_b 
        lowE = (E_b)**alpha * np.exp(-E_b / E_0_b)
        highE = ((alpha - beta) * E_0_b)**(alpha - beta) * np.exp(beta - alpha) * (E_b)**beta
    return np.where(cond, lowE, highE)

def fred(t, tau_1, tau_2):
    """
    FRED light curve function for temporal evolution.

    Args:
        t (array): Time values for the light curve.
        tau_1 (float): Parameter controlling the rise of the light curve.
        tau_2 (float): Parameter controlling the decay of the light curve.

    Returns:
        array: FRED light curve evaluated at each time point.
    """
    return np.array(np.exp(2 * (tau_1 / tau_2) ** 0.5) / np.exp(tau_1 / t + t / tau_2))

def impulse(t, t_peak, width=1e-3):
    """
    Models an impulse function centered at `t_peak` with a given width.
    
    Parameters:
        t (array): Time grid.
        t_peak (float): Time at which the impulse occurs.
        width (float): Width of the Gaussian impulse (small value for sharpness).
    
    Returns:
        array: The impulse-like function at each time step.
    """
    # Narrow Gaussian to simulate an impulse (width determines sharpness)
    return np.exp(-0.5 * ((t - t_peak) / width) ** 2)

def angular_d(theta_1, theta_2, phi_1, phi_2):
    """
    Angular distance between two points on a sphere.

    Args:
        theta_1, theta_2 (float): Polar angles (in radians) for the two points.
        phi_1, phi_2 (float): Azimuthal angles (in radians) for the two points.

    Returns:
        float: Angular distance between the two points.
    """
    return np.arccos(np.round(np.sin(np.pi/2 - theta_1) * np.sin(np.pi/2 - theta_2) + np.cos(np.pi/2 - theta_1) * np.cos(np.pi/2 - theta_2) * np.cos(phi_1 - phi_2), 12))

def spherical_to_cartesian(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to Cartesian coordinates (x, y).

    Args:
        theta (array): Polar angle (in radians).
        phi (array): Azimuthal angle (in radians).

    Returns:
        tuple: Cartesian coordinates (x, y) corresponding to the spherical coordinates.
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    return x, y

def lg11(eps):
    """
    Liang-Ghirlanda (2011) relation for Lorentz Factor.

    Args:
        eps (array): Energy per solid angle values at each theta x phi.

    Returns:
        array: Lorentz factor.
    """
    return 182 * (eps / 1e52)**0.25 + 1

def nearest_coord(theta, phi, theta_los, phi_los):
    """
    Find the nearest grid coordinates (theta, phi) to a given line-of-sight (theta_los, phi_los).

    Args:
        theta (array): Array of theta grid values.
        phi (array): Array of phi grid values.
        theta_los (float): Line-of-sight polar angle.
        phi_los (float): Line-of-sight azimuthal angle.

    Returns:
        tuple: Indices (theta_i, phi_i) corresponding to the nearest grid point.
    """
    theta_i = np.abs(theta[0, :] - theta_los).argmin()
    phi_i = np.abs(phi[:, 0] - phi_los).argmin()
    return theta_i, phi_i

def int_spec(E, N_E, E_min=None, E_max=None):
    """
    Integrate spectrum over energy range defined by E_min and E_max.

    Args:
        E (array): Energy values for spectrum.
        N_E (array): Band spectrum values.
        E_min (float, optional): Minimum energy for integration.
        E_max (float, optional): Maximum energy for integration.

    Returns:
        float: Integral of spectrum over specified energy range.
    """
    E = np.asarray(E)
    N_E = np.asarray(N_E)

    # Create mask over energy range
    mask = np.ones_like(E, dtype=bool)
    if E_min is not None:
        mask &= E >= E_min
    if E_max is not None:
        mask &= E <= E_max

    # Restrict E and N_E to the masked energy range
    E_masked = E[mask]
    if N_E.ndim == 1:
        N_E_masked = N_E[mask]
        return np.trapezoid(N_E_masked * E_masked, E_masked)
    elif N_E.ndim == 3:
        N_E_masked = N_E[..., mask]
        E_masked_broadcast = E_masked[np.newaxis, np.newaxis, :]
        return np.trapezoid(N_E_masked * E_masked_broadcast, E_masked, axis=-1)
    else:
        raise ValueError("N_E must be 1D or 3D array.")

    
def int_lc(t, L):
    """
    Integrate light curve over time.

    Args:
        t (array): Time values for light curve.
        L (array): Light curve values.

    Returns:
        float: The integral of light curve over time.
    """
    return np.trapezoid(L, t)

def interp_lc(t, L):
    """
    Interpolate light curve over a common time grid.

    Args:
        t (array): Time values for light curve.
        L (array): Light curve values.

    Returns:
        tuple: Interpolated time grid and total interpolated light curve.
    """
    t_common = np.geomspace(1e-3, 1e6, 1000)
    L_total = np.zeros_like(t_common)

    t_flat = t.reshape(-1, t.shape[-1])
    L_flat = L.reshape(-1, L.shape[-1])

    for t_ij, L_ij in zip(t_flat, L_flat):
        if np.all(np.isfinite(t_ij)) and np.any(L_ij > 0):
            mask = (L_ij > 0) & np.isfinite(t_ij)
            t_valid = t_ij[mask]
            L_valid = L_ij[mask]
            
            if len(t_valid) > 1:
                L_interp = np.interp(t_common, t_valid, L_valid, left=0.0, right=0.0)
                L_total += L_interp

    return t_common, L_total

def interp_spec(E, N):
    """
    Interpolate spectra over a common energy grid and sum contributions.

    Args:
        E (array): Energy values for spectra (e.g. in keV), shape (..., n_E).
        N (array): Photon number spectra (e.g. dN/dE), same shape as E.

    Returns:
        tuple: Interpolated energy grid and total interpolated spectrum.
    """
    E_common = np.geomspace(1e2, 1e6, 1000)  # adjust as needed for your energy band
    N_total = np.zeros_like(E_common)

    E_flat = E.reshape(-1, E.shape[-1])
    N_flat = N.reshape(-1, N.shape[-1])

    for E_ij, N_ij in zip(E_flat, N_flat):
        if np.all(np.isfinite(E_ij)) and np.any(N_ij > 0):
            mask = (N_ij > 0) & np.isfinite(E_ij)
            E_valid = E_ij[mask]
            N_valid = N_ij[mask]

            if len(E_valid) > 1:
                N_interp = np.interp(E_common, E_valid, N_valid, left=0.0, right=0.0)
                N_total += N_interp

    return E_common, N_total


def save_data(jet, wind, theta_los, phi_los, path='./', model_id=0):
    """
    Save time series data of jet and wind to a CSV file.

    Args:
        jet (object): Jet object.
        wind (object): Wind object.
        theta_los (float): Line-of-sight polar angle in radians.
        phi_los (float): Line-of-sight azimuthal angle in radians.
        path (str): Directory path to save the CSV file.
        model_id (int): Determines how to handle data (1-4).

    Returns:
        None
    """
    with open(path + '{}_data.csv'.format(int(round(np.rad2deg(theta_los)))), mode='w', newline='') as file:
        writer = csv.writer(file)
        if model_id == 1 or model_id == 2:
            writer.writerow(['jet_t', 'jet_L_gamma', 'jet_L_X', 'wind_t', 'wind_L_X'])
            for i in range(jet.t.shape[0]):
                writer.writerow([jet.t[i], jet.L_gamma_tot[i], jet.L_X_tot[i], wind.engine.t[i], wind.L_X_tot[i]])
        elif model_id == 3 or model_id == 4: 
            writer.writerow(['jet_t', 'jet_L_gamma', 'jet_L_X',])
            for i in range(jet.t.shape[0]):
                writer.writerow([jet.t[i], jet.L_gamma_tot[i], jet.L_X_tot[i]])
    return

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
def obs_grid(eps, e_iso_grid, amati_index, e_1=0.3e3, e_2=10e3):
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

    # Calculate the peak and cutoff energy based on the Amati relation
    E_p = E_p_0 * (e_iso_grid / e_iso_grid[0])**amati_index
    # E_p = 1e5 * 10**(0.83 + 0.41 * np.log10(e_iso_grid / 1e51))
    # [print(f'E_p: {ep}, E_iso: {eiso}') for e p, eiso in zip(E_p, e_iso_grid)]

    E_0 = E_p / (2 + alpha)

    # Define the energy grid for the spectrum integration
    E = np.geomspace(1e2, 1e6, 1000)

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
    a_1 = 0.05
    a_2 = 0.15

    # Generate FRED light curve
    t = np.geomspace(1e-3, 1e3, 1000)
    L = fred(t, a_1, a_2)
    
    # t_peak = 1e-2
    # t = np.geomspace(t_peak/10, t_peak*10, 100)
    # L = impulse(t, t_peak, width=1e-3) 

    # Normalize time-integrated Luminosity
    S_unit = int_lc(t, L)
    A_lc = S / S_unit
    L_scaled = A_lc[..., np.newaxis] * L 

    return E, N_E_norm, t, L_scaled, S

def e_iso_grid(theta, phi, g, eps, dOmega):
    """
    Computes the isotropic equivalent energy (E_iso) for a grid of angles.
    """

    E_iso_grid = np.zeros_like(theta[0])

    # Loop over each theta (phi-independent)
    for i_theta in range(len(theta[0])):
        theta_los = theta[0, i_theta]
        D_on = doppf(g, 0)
        D_off = doppf(g, angular_d(theta[0, i_theta], theta, phi[0, 0], phi))
        R_D = D_off / D_on        

        # Energy observed at this grid point
        E_iso = 4 * np.pi * np.sum(eps[eps > 0] * R_D[eps > 0]**3 * dOmega[eps > 0]) / np.sum(R_D[eps > 0]**2 * dOmega[eps > 0])
        E_iso_grid[i_theta] = E_iso  # Store the calculated E_iso for each grid point
    return E_iso_grid