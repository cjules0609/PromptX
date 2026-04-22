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
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from .const import *

def z2d_L(z):
    return cosmo.luminosity_distance(z).to('cm').value

def gamma2beta(gamma):
    """
    Convert Lorentz factor to velocity factor.

    Args:
        gamma (float): Lorentz factor, must be greater than or equal to 1.

    Returns:
        float: Velocity factor (beta), between 0 and 1.
    """
    return np.sqrt(1 - 1 / gamma**2)

def beta2gamma(beta):
    """
    Convert velocity factor to Lorentz factor.

    Args:
        beta (float): Velocity factor, must be between 0 and 1.

    Returns:
        float: Lorentz factor (gamma), greater than or equal to 1.

    Raises:
        ValueError: If beta is not in the range [0, 1).
    """
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
    return np.exp(-((x - mu)**2 / (2 * sigma**2)))

def powerlaw(x, theta_jet, k):
    """
    Power-law jet profile with a flat core.

    Args:
        x (array): Input angles (radians).
        theta_jet (float): Core angle (radians) where profile is flat.
        k (float): Power-law exponent.

    Returns:
        array: Normalized profile, equals 1 at theta=0..theta_jet,
               and (theta/theta_jet)^(-k) beyond.
    """
    return np.where(x <= theta_jet, 1.0, (x / theta_jet) ** (-k))

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
    return np.exp((2 * (tau_1 / tau_2) ** 0.5) - (tau_1 / t + t / tau_2))

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
    Angular distance between two points on a full sphere.

    Args:
        theta_1, theta_2 (float): Polar angles [0, π] in radians.
        phi_1, phi_2 (float): Azimuthal angles [0, 2π] in radians.

    Returns:
        float: Angular distance in radians between the two points.
    """
    return np.arccos(
        np.clip(
            np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1 - phi_2) +
            np.cos(theta_1) * np.cos(theta_2),
            -1.0, 1.0
        )
    )

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

def lg11(e_iso, theta=None, theta_cut=None):
    """
    Liang-Ghirlanda (2011) relation for Lorentz Factor.

    Args:
        e_iso (array): Observed isotropic-equivalent energy at each theta x phi.

    Returns:
        array: Lorentz factor.
    """
    # Radiative efficiency
    eta_gamma = 0.01
    Gamma_0 = 200

    Gamma = (Gamma_0 / (1 - eta_gamma)) * (e_iso / 1e52)**0.25 + 1
    return np.where((theta is not None) & (theta_cut is not None) & (np.abs(np.cos(theta)) < np.cos(theta_cut)), 1.0, Gamma)

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
    theta_i = np.abs(theta[:, 0] - theta_los).argmin()
    phi_i = np.abs(phi[0, :] - phi_los).argmin()
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

def interp_lc(t, L, t_common=None):
    """
    Interpolate light curve over a common time grid.

    Args:
        t (array): Time values for light curve.
        L (array): Light curve values.

    Returns:
        tuple: Interpolated time grid and total interpolated light curve.
    """
    if t_common is None:
        t_common = np.geomspace(1e-3, 1e6, 1000)
    
    # Flatten all leading dimensions to (n_curves, n_time)
    t_flat = t.reshape(-1, t.shape[-1])
    L_flat = L.reshape(-1, L.shape[-1])
    
    # Initialize total array
    L_total = np.zeros_like(t_common)
    
    # Vectorized interpolation using list comprehension
    L_total = np.sum([np.interp(t_common, t_flat[i], L_flat[i], left=0.0, right=0.0)
                      for i in range(t_flat.shape[0])], axis=0)
    
    return t_common, L_total

def interp_spec(E, N, E_common=None):
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
    
    N_total = np.sum([np.interp(E_common, E_flat[i], N_flat[i], left=0.0, right=0.0)
                      for i in range(E_flat.shape[0])], axis=0)
    
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
    # Uniform parameter u in [0,1]
    u = np.linspace(0, 1, n_theta)
    alpha = 3
    
    # Symmetric tanh mapping
    theta = 0.5*np.pi * (1 + np.tanh(alpha*(2*u-1)) / np.tanh(alpha))    

    phi = np.linspace(phi_bounds[0], phi_bounds[1], n_phi)
    
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    return TH, PH

def gamma_grid(g0, theta, phi, struct='tophat', theta_jet=np.deg2rad(5), cutoff=None, **kwargs):
    """
    Generates grid of Lorentz factors based on given structure type.

    Args:
        g0 (float): Lorentz factor normalization.
        theta (2D array): Meshgrid of polar angles (radians).
        struct (str or callable): 'tophat', 'gaussian', 'powerlaw', or a custom function.
        theta_jet (float, optional): Core angle (radians) for Gaussian or power-law. Defaults to 5 deg.
        cutoff (float, optional): Angle beyond which Gamma is set to 1.
        kwargs: Additional parameters like 'k' for power-law exponent.

    Returns:
        2D array: Lorentz factor grid.
    """

    struct_map = {1: 'tophat', 2: 'gaussian', 3: 'powerlaw'}
    if callable(struct):
        struct = struct
    elif struct in struct_map:
        struct = struct_map[struct]
    elif isinstance(struct, str) and struct.lower() in struct_map.values():
        struct = struct.lower()

    if callable(struct):
        g = g0 * struct(theta, phi, theta_jet=theta_jet, **kwargs)
    elif struct == 'tophat':
        g = g0 * (theta <= theta_jet).astype(float)
    elif struct == 'gaussian':
        g = g0 * gaussian(theta, theta_jet)
    elif struct == 'powerlaw':
        g = 1 + (g0 - 1) / (1 + (theta/theta_jet)**kwargs['k'])
    else:
        raise ValueError("struct must be 'tophat', 'gaussian', 'powerlaw', or a callable")

    g[g < 1] = 1
    if cutoff:
        g[np.abs(np.cos(theta)) < np.cos(cutoff)] = 1

    return g

def eps_grid(eps0, theta, phi, struct='tophat', theta_jet=np.radians(5), cutoff=np.radians(90), **kwargs):
    """
    Generates grid of energy per solid angle (eps) with optional counter jet.

    Parameters:
        eps0 (float): normalization
        theta, phi (ndarray): grid of polar and azimuthal angles
        struct (str or callable): 'tophat', 'gaussian', 'powerlaw', or a callable
        theta_jet (float): core angle for Gaussian/PL
        k (float): power-law index if struct='powerlaw'
        cutoff (float): cutoff angle beyond which eps=0
        **kwargs: additional args passed to callable struct
    """

    struct_map = {1: 'tophat', 2: 'gaussian', 3: 'powerlaw'}
    if callable(struct):
        struct = struct
    elif struct in struct_map:
        struct = struct_map[struct]
    elif isinstance(struct, str) and struct.lower() in struct_map.values():
        struct = struct.lower()

    if callable(struct):
        eps = eps0 * struct(theta, phi, theta_jet=theta_jet, **kwargs)
    elif struct == 'tophat':
        eps = eps0 * (theta <= theta_jet).astype(float)
    elif struct == 'gaussian':
        eps = eps0 * gaussian(theta, theta_jet)
    elif struct == 'powerlaw':
        eps = eps0 * powerlaw(theta, theta_jet, kwargs['k'])
    else:
        raise ValueError("struct must be 'tophat', 'gaussian', 'powerlaw', or a callable")

    if cutoff:
        eps[np.abs(np.cos(theta)) < np.cos(cutoff)] = 0
    return eps


def obs_grid(eps, e_iso_grid, amati_a=0.41, amati_b=0.83, e_1=0.3e3, e_2=10e3, tau_1=0.1, tau_2=0.35):
    """
    Computes the spectrum and light curve observed by an observer on a grid.

    This function calculates the observed spectrum and light curve at each grid based on the intrinsic energy per solid angle grid 
    (eps) and observed isotropic energy grid (e_iso_grid). It uses the Amati relation to determine the peak energy 
    (E_p) of the Band spectra, each normalized such that the integral equals eps. The light curve at each grid is modeled 
    using a FRED function, each normalized such that the integral equals eps.

    Args:
        eps (2D array): Energy per solid angle grid values.
        e_iso_grid (2D array): Isotropic energy grid values for each grid point.
        amati_a (float): Amati relation slope used to calculate the peak energy.
        amati_b (float): Amati relation intercept used to calculate the peak energy.
        e_1 (float): Minimum energy for the spectrum integration, default is 0.3 keV.
        e_2 (float): Maximum energy for the spectrum integration, default is 10 keV.

    Returns:
        tuple: A tuple containing the following arrays:
            - E (array): Energy grid used for the spectrum integration.
            - N_E_norm (array): Normalized Band spectrum.
            - t (array): Time grid adjusted to the observer frame for the light curve.
            - L_scaled (array): Scaled light curve.
            - S (array): Integrated spectrum over the detector energy band.
    """
    # SPECTRUM
    # Reference Band function parameters
    alpha, beta = -1, -2.3
    E_p = 1e5 * 10**(amati_a * np.log10(e_iso_grid / 1e51) + amati_b)

    # Calculate the peak and cutoff energy based on the Amati relation
    # E_p = E_p_0 * (e_iso_grid / e_iso_grid[0])**amati_index
    E_0 = E_p / (2 + alpha)

    # Define the energy grid for the spectrum integration
    E = np.geomspace(1e2, 1e8, 1000)

    # Compute the unnormalized Band energy spectrum, EN(E)
    EN_E = E * band(E, alpha, beta, E_0)

    # Normalize spectrum to eps
    eps_unit = int_spec(E, EN_E, E_min=1e3, E_max=10e6)
    mask = eps_unit > 0
    A_spec = np.zeros_like(eps)
    A_spec[mask] = eps[mask] / eps_unit[mask]
    EN_E_norm = A_spec[..., np.newaxis] * EN_E 

    # Integrate spectra of emitting regions over detector energy band
    S = int_spec(E, EN_E_norm, E_min=e_1, E_max=e_2)
    S = np.nan_to_num(S, nan=0.0)

    # Generate FRED light curve
    t = np.geomspace(1e-3, 1e4, 1000)
    L = fred(t, tau_1, tau_2)

    # Normalize time-integrated Luminosity
    S_unit = int_lc(t, L)
    A_lc = S / S_unit
    L_scaled = A_lc[..., np.newaxis] * L

    return E, EN_E_norm, t, L_scaled, S

def calc_e_iso_grid(theta, phi, g, eps, theta_cut, dOmega):
    n_theta = theta.shape[0]
    E_iso_grid = np.zeros(n_theta)
    D_on = doppf(g, 0)
    dOmega_sum = np.sum(dOmega)

    th = theta[:, 0]
    ph = phi[0, :]

    cos_theta_v = (
        np.cos(th)[:, None, None] * np.cos(th)[None, :, None] +
        np.sin(th)[:, None, None] * np.sin(th)[None, :, None] *
        np.cos(ph)[None, None, :]
    )
    theta_v = np.arccos(np.clip(cos_theta_v, -1.0, 1.0))

    mask_jet = theta_v <= theta_cut
    mask_counter = theta_v >= (np.pi - theta_cut)

    R_D_jet = doppf(g[None, :, :], theta_v) / D_on[None, :, :]
    R_D_counter = doppf(g[None, :, :], np.pi - theta_v) / D_on[None, :, :]

    eps_b = eps[None, :, :]
    dOmega_b = dOmega[None, :, :]

    E_iso_jet = 2 * np.pi * np.sum(eps_b * R_D_jet**3 * dOmega_b * mask_jet, axis=(1, 2)) / (dOmega_sum)
    E_iso_counter = 2 * np.pi * np.sum(eps_b * R_D_counter**3 * dOmega_b * mask_counter, axis=(1, 2)) / (dOmega_sum)

    E_iso_grid = (E_iso_jet + E_iso_counter)

    E_iso_grid = np.minimum(E_iso_grid, E_iso_grid[0])
    E_iso_grid = np.maximum(E_iso_grid, 1e-30)

    E_iso_grid = np.tile(E_iso_grid[:, np.newaxis], (1, phi.shape[1]))

    return E_iso_grid

def profile_interp(orig_theta, orig_phi, orig_profile, theta_rot, phi_rot, method='nearest'):
    """
    Interpolate a profile defined on spherical coordinates onto a rotated grid
    using 3D Cartesian coordinates.
    """
    # --- Step 1: Convert original cell centers to Cartesian ---
    x = np.sin(orig_theta) * np.cos(orig_phi)
    y = np.sin(orig_theta) * np.sin(orig_phi)
    z = np.cos(orig_theta)
    points_xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # --- Step 2: Convert rotated grid to Cartesian ---
    x_rot = np.sin(theta_rot) * np.cos(phi_rot)
    y_rot = np.sin(theta_rot) * np.sin(phi_rot)
    z_rot = np.cos(theta_rot)
    points_rot_xyz = np.column_stack([x_rot.ravel(), y_rot.ravel(), z_rot.ravel()])
    
    # --- Step 3: Interpolate in 3D ---
    rotated_profile = griddata(points_xyz, orig_profile.ravel(), points_rot_xyz, method=method, fill_value=0.0)
    
    return rotated_profile.reshape(theta_rot.shape)

def rotate_spherical(theta, phi, theta_target, phi_target):
    """
    Rotate spherical coordinates so that the north pole aligns with (theta_target, phi_target).

    Parameters
    ----------
    theta, phi : array_like
        Input spherical coordinates (can be scalars, 1D arrays, or 2D meshgrids).
    theta_target, phi_target : float
        Target spherical coordinates to rotate the north pole onto.

    Returns
    -------
    theta_rot, phi_rot : array_like
        Rotated spherical coordinates (same shape as input).
    """

    # --- Step 1: Rotation matrix ---
    target_vec = np.array([
        np.sin(theta_target) * np.cos(phi_target),
        np.sin(theta_target) * np.sin(phi_target),
        np.cos(theta_target)
    ])
    north_pole = np.array([0.0, 0.0, 1.0])

    rot_axis = np.cross(north_pole, target_vec)
    axis_norm = np.linalg.norm(rot_axis)
    if axis_norm < 1e-12:
        R = np.eye(3)
    else:
        rot_axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(north_pole, target_vec), -1.0, 1.0))
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                      [rot_axis[2], 0, -rot_axis[0]],
                      [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # --- Step 2: Cartesian from spherical ---
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    xyz = np.stack([x, y, z], axis=-1)

    # --- Step 3: Rotate ---
    xyz_rot = xyz @ R.T
    x_rot, y_rot, z_rot = xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2]

    # --- Step 4: Back to spherical ---
    theta_rot = np.arccos(np.clip(z_rot, -1.0, 1.0))
    phi_rot = np.arctan2(y_rot, x_rot) % (2*np.pi)

    return theta_rot, phi_rot

def calc_t90(t, L, z=1, windows=[0.064,0.256,1,4,8], F_lim_1s=2.8e-8):

    if len(t) < 2:
        return np.nan, False, 0.0

    d_L = cosmo.luminosity_distance(z).to(u.cm).value

    flux_t = L / (4 * np.pi * d_L**2)

    dt = np.diff(t)
    if np.any(dt <= 0):
        return np.nan, False, 0.0

    trap = (flux_t[:-1] + flux_t[1:]) * 0.5 * dt
    cum_fluence = np.concatenate([[0], np.cumsum(trap)])

    total_fluence = cum_fluence[-1]
    if total_fluence <= 0:
        return np.nan, False, 0.0

    t5  = np.interp(0.05 * total_fluence, cum_fluence, t)
    t95 = np.interp(0.95 * total_fluence, cum_fluence, t)
    T90 = t95 - t5

    detected = False
    F_peak_global = 0

    for dt_trigger in windows:

        F_lim = F_lim_1s * np.sqrt(1.0 / dt_trigger)

        for i in range(len(t)):

            t_end = t[i] + dt_trigger
            j = np.searchsorted(t, t_end)
            j = min(j, len(t)-1)

            if j <= i:
                continue

            flu = cum_fluence[j] - cum_fluence[i]
            F_avg = flu / dt_trigger

            if F_avg > F_peak_global:
                F_peak_global = F_avg

            if F_avg >= F_lim:
                detected = True
    T90 = np.nan if not detected else T90
    return T90, detected, F_peak_global