# ============================================================================= #
#                  ____  ____   __   _  _  ____  ____  _  _                     #
#                 (  _ \(  _ \ /  \ ( \/ )(  _ \(_  _)( \/ )                    #
#                  ) __/ )   /(  O )/ \/ \ ) __/  )(   )  (                     #
#                 (__)  (__\_) \__/ \_)(_/(__)   (__) (_/\_)                    #
#                                                                               #
# ============================================================================= #
#   PromptX - Prompt X-ray emission modeling of relativistic outflows           #
#   Version 0.2                                                                 #
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
    """Convert Lorentz factor to dimensionless velocity."""
    return np.sqrt(1 - 1 / gamma**2)

def beta2gamma(beta):
    """Convert dimensionless velocity to Lorentz factor."""
    return 1 / np.sqrt(1 - beta**2)

def gaussian(x, sigma, mu=0):
    """Evaluate a Gaussian profile."""
    return np.exp(-((x - mu)**2 / (2 * sigma**2)))

def powerlaw(x, theta_jet, k):
    """Return a flat-core power-law angular profile."""
    return np.where(x <= theta_jet, 1.0, (x / theta_jet) ** (-k))

def doppf(g, theta):
    """Compute the relativistic Doppler factor."""
    return 1 / (g * (1 - np.sqrt(1 - 1 / g / g) * np.cos(theta)))

def band(E, alpha, beta, E_0):
    """Evaluate the Band spectrum on a scalar or grid-valued cutoff energy."""
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
    """Return a narrow Gaussian pulse centered at ``t_peak``."""
    return np.exp(-0.5 * ((t - t_peak) / width) ** 2)

def angular_d(theta_1, theta_2, phi_1, phi_2):
    """Return the spherical angular separation between two directions."""
    return np.arccos(
        np.clip(
            np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1 - phi_2) +
            np.cos(theta_1) * np.cos(theta_2),
            -1.0, 1.0
        )
    )

def spherical_to_cartesian(theta, phi):
    """Project spherical coordinates onto Cartesian ``x`` and ``y`` axes."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    return x, y

def lg11(e_iso, theta=None, theta_cut=None):
    """Map isotropic-equivalent energy to Lorentz factor using LG11."""
    eta_gamma = 0.01
    Gamma_0 = 200

    Gamma = (Gamma_0 / (1 - eta_gamma)) * (e_iso / 1e52)**0.25 + 1
    return np.where((theta is not None) & (theta_cut is not None) & (np.abs(np.cos(theta)) < np.cos(theta_cut)), 1.0, Gamma)

def nearest_coord(theta, phi, theta_los, phi_los):
    """Return the grid indices nearest to a line of sight."""
    theta_i = np.abs(theta[:, 0] - theta_los).argmin()
    phi_i = np.abs(phi[0, :] - phi_los).argmin()
    return theta_i, phi_i

def int_spec(E, N_E, E_min=None, E_max=None):
    """Integrate an energy-weighted spectrum over an optional bandpass."""
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
    """Integrate a light curve over time."""
    return np.trapezoid(L, t)

def interp_lc(t, L, t_common=None):
    """Interpolate and sum light curves onto a common time grid."""
    if t_common is None:
        t_common = np.geomspace(1e-3, 1e10, 1000)
    
    t_flat = t.reshape(-1, t.shape[-1])
    L_flat = L.reshape(-1, L.shape[-1])
    L_total = np.zeros_like(t_common)
    L_total = np.sum([np.interp(t_common, t_flat[i], L_flat[i], left=0.0, right=0.0)
                      for i in range(t_flat.shape[0])], axis=0)
    
    return t_common, L_total

def interp_spec(E, N, E_common=None):
    """Interpolate and sum spectra onto a common energy grid."""
    E_common = np.geomspace(1e2, 1e6, 1000)  # adjust as needed for your energy band
    N_total = np.zeros_like(E_common)

    E_flat = E.reshape(-1, E.shape[-1])
    N_flat = N.reshape(-1, N.shape[-1])
    
    N_total = np.sum([np.interp(E_common, E_flat[i], N_flat[i], left=0.0, right=0.0)
                      for i in range(E_flat.shape[0])], axis=0)
    
    return E_common, N_total

def save_data(jet, wind, theta_los, phi_los, path='./', model_id=0):
    """Write jet and optional wind light-curve data to a CSV file."""
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
    """Build the spherical edge grid used by the outflow meshes."""
    u = np.linspace(0, 1, n_theta)
    alpha = 3
    theta = 0.5*np.pi * (1 + np.tanh(alpha*(2*u-1)) / np.tanh(alpha))    

    phi = np.linspace(phi_bounds[0], phi_bounds[1], n_phi)
    
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    return TH, PH

def gamma_grid(g0, theta, phi, struct='tophat', theta_jet=np.deg2rad(5), cutoff=None, **kwargs):
    """Build a Lorentz-factor grid for the requested angular structure."""

    struct_map = {0: 'tophat', 1: 'tophat', 2: 'gaussian', 3: 'powerlaw'}
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
    """Build an energy or luminosity-per-solid-angle grid."""

    struct_map = {0: 'tophat', 1: 'tophat', 2: 'gaussian', 3: 'powerlaw'}
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


# def radiation_field(model, eps, e_iso_grid, E, t):

#     rad = Radiation(model)

#     EN_E = rad.spectrum(E, eps, e_iso_grid)
#     L = rad.light_curve(t, None)  # depends on your design

#     return EN_E, L

def observed_emission(model, eps, e_iso_grid, E, t,
             amati_a=0.41, amati_b=0.83,
             e_1=0.3e3, e_2=10e3,
             tau_1=0.1, tau_2=0.35):
    """Construct per-cell spectra, light curves, and band-limited fluence."""
    class _ObservedEmissionOutflow:
        pass

    outflow = _ObservedEmissionOutflow()
    outflow.eps = eps
    outflow.e_iso_grid = e_iso_grid

    NE_kernel = model.build_spectrum_kernel(E, outflow)
    L_kernel = model.build_light_curve_kernel(t, outflow)

    eps_unit = int_spec(E, E * NE_kernel, E_min=1e3, E_max=10e6)
    mask = eps_unit > 0
    A_spec = np.zeros_like(eps)
    A_spec[mask] = eps[mask] / eps_unit[mask]
    EN_E = A_spec[..., np.newaxis] * E * NE_kernel

    S = int_spec(E, EN_E, E_min=e_1, E_max=e_2)
    S = np.nan_to_num(S, nan=0.0)

    S_unit = int_lc(t, L_kernel)
    L = (S / S_unit)[..., np.newaxis] * L_kernel

    return EN_E, L, S

def calc_e_iso_grid(theta, phi, g, eps, theta_cut, dOmega):
    """Compute isotropic-equivalent energy as a function of viewing angle."""
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

def calc_L_iso_grid(theta, phi, g, dL_dOmega, theta_cut, dOmega):
    """Compute isotropic-equivalent luminosity as a function of viewing angle."""
    n_theta = theta.shape[0]
    dL_dOmega_grid = np.zeros(n_theta)
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

    eps_b = dL_dOmega[None, :, :]
    dOmega_b = dOmega[None, :, :]

    dL_dOmega_jet = 2 * np.pi * np.sum(eps_b * R_D_jet**4 * dOmega_b * mask_jet, axis=(1, 2)) / (dOmega_sum)
    dL_dOmega_counter = 2 * np.pi * np.sum(eps_b * R_D_counter**4 * dOmega_b * mask_counter, axis=(1, 2)) / (dOmega_sum)

    dL_dOmega_grid = (dL_dOmega_jet + dL_dOmega_counter)

    dL_dOmega_grid = np.minimum(dL_dOmega_grid, dL_dOmega_grid[0])
    dL_dOmega_grid = np.maximum(dL_dOmega_grid, 1e-30)

    dL_dOmega_grid = np.tile(dL_dOmega_grid[:, np.newaxis], (1, phi.shape[1]))

    return dL_dOmega_grid

def profile_interp(orig_theta, orig_phi, orig_profile, theta_rot, phi_rot, method='nearest'):
    """Interpolate a spherical profile onto a rotated spherical grid."""
    x = np.sin(orig_theta) * np.cos(orig_phi)
    y = np.sin(orig_theta) * np.sin(orig_phi)
    z = np.cos(orig_theta)
    points_xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    x_rot = np.sin(theta_rot) * np.cos(phi_rot)
    y_rot = np.sin(theta_rot) * np.sin(phi_rot)
    z_rot = np.cos(theta_rot)
    points_rot_xyz = np.column_stack([x_rot.ravel(), y_rot.ravel(), z_rot.ravel()])
    
    rotated_profile = griddata(points_xyz, orig_profile.ravel(), points_rot_xyz, method=method, fill_value=0.0)
    
    return rotated_profile.reshape(theta_rot.shape)

def rotate_spherical(theta, phi, theta_target, phi_target):
    """
    Rotate spherical coordinates so the pole points at a target direction.
    """
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

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    xyz = np.stack([x, y, z], axis=-1)

    xyz_rot = xyz @ R.T
    x_rot, y_rot, z_rot = xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2]

    theta_rot = np.arccos(np.clip(z_rot, -1.0, 1.0))
    phi_rot = np.arctan2(y_rot, x_rot) % (2*np.pi)

    return theta_rot, phi_rot

def calc_t90(t, L, z=1, windows=[0.064,0.256,1,4,8], F_lim_1s=2.8e-8):
    """Estimate T90 and trigger detectability from a bolometric light curve."""

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
