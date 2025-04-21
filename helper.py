import numpy as np
import csv

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

def save_data(jet, wind, theta_los, path='./', model_id=0):
    """
    Save time series data of jet and wind to a CSV file.

    Args:
        jet (object): Jet object.
        wind (object): Wind object.
        theta_los (float): Line-of-sight angle in radians.
        path (str): Directory path to save the CSV file.
        model_id (int): Determines how to handle data (1-4).

    Returns:
        None
    """
    with open(path + '{}_data.csv'.format(int(round(np.rad2deg(theta_los)))), mode='w', newline='') as file:
        writer = csv.writer(file)
        if model_id == 1 or model_id == 2:
            writer.writerow(['jet_t', 'jet_L_gamma', 'jet_L_X', 'wind_t', 'wind_L_X'])
            print(jet.t.shape)
            for i in range(jet.t.shape[0]):
                writer.writerow([jet.t[i], jet.L_gamma_tot[i], jet.L_X_tot[i], wind.engine.t[i], wind.L_X_tot[i]])
        elif model_id == 3 or model_id == 4: 
            writer.writerow(['jet_t', 'jet_L_gamma', 'jet_L_X',])
            for i in range(jet.t.shape[0]):
                writer.writerow([jet.t[i], jet.L_gamma_tot[i], jet.L_X_tot[i]])
    return