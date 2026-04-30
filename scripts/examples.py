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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import os

from promptx.const import c
from promptx.Jet import Jet
from promptx.Wind import Wind
from promptx.Engine import Engine
from promptx.Observer import Observer
from promptx.Opacity import Opacity
from promptx.Radiation import Radiation
from promptx.RadModels import Phenomenological, SpindownWind

plt.rcParams.update({'font.size': 12})

def plot_lc(jet, jet_obs, wind, wind_obs, path='./out/', model_id=0):
    """Plot observed jet and optional wind light curves."""

    fig_lc, ax_lc = plt.subplots()

    t_jet, L_jet = jet_obs.light_curve(E_min=1e3, E_max=10e6)

    ax_lc.plot(t_jet, L_jet, lw=1, c='r', ls='-', label='Jet')
    
    if model_id in [1, 2]:
        t_wind, L_wind = wind_obs.light_curve()
        ax_lc.plot(t_wind, L_wind, lw=1, c='b', label='Wind')

    ax_lc.set_xlabel(r'$t$ [s]')
    ax_lc.set_ylabel(r'Luminosity [erg s$^{-1}$]')
    ax_lc.set_xscale('log')
    ax_lc.set_yscale('log')
    ax_lc.set_ylim([6e37, 2e54])
    ax_lc.set_xlim([1e-3, 1e8])
    ax_lc.legend()
    ax_lc.grid()

    plt.savefig(path + 'lc.png', dpi=300)
    plt.show()

def plot_spec(jet, jet_obs, path='./out/', model_id=0):
    """Plot the observed jet spectrum."""

    fig_spec, ax_spec = plt.subplots()

    E = jet_obs.radiation.E
    ax_spec.plot(E, E * jet_obs.EN_E, c='r', lw=1)

    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')
    ax_spec.set_xlabel('E [eV]')
    ax_spec.set_ylabel(r'$E^2 N(E)$')
    ax_spec.set_ylim([1e40, 1e46])
    ax_spec.grid()

    plt.savefig(path + 'spec.png', dpi=300)
    plt.show()

def plot_jet_lc_obs(jet_obs, theta_los, phi_los, path='./out/'):
    """Plot patch gamma-ray light curves and the total observed gamma-ray signal."""

    plt.rcParams.update({'font.size': 12})
    jet = jet_obs.outflow
    los_coord = [np.abs(jet.theta[:, 0] - theta_los).argmin(), np.abs(jet.phi[0, :] - phi_los).argmin()]

    n_colors = 256
    cut = np.rad2deg(jet.theta_cut) / 90
    n_cut = int(n_colors * cut)

    cmap_colors = cm.plasma(np.linspace(0, 1, n_cut))
    gray_colors = np.tile(np.array([[0.8, 0.8, 0.8, 1.0]]), (n_colors - n_cut, 1))

    colors = np.vstack([cmap_colors, gray_colors])
    custom_cmap = LinearSegmentedColormap.from_list("plasma+grey", colors)

    norm = mcolors.Normalize(vmin=0, vmax=90)
    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    t_gamma, L_gamma = jet_obs.light_curve(E_min=1e3, E_max=10e6)

    stride = max(1, jet.theta.shape[0] // 40)
    for theta_i in range(0, jet.theta.shape[0], stride):
        theta_deg = np.rad2deg(jet.theta[theta_i, 0])
        color = custom_cmap(norm(theta_deg))
        patch_lc = jet_obs.L_t_obs[theta_i, 0] * 4 * np.pi / np.sum(jet.dOmega)
        ax.plot(jet_obs.t_obs[theta_i, 0], patch_lc, color=color, lw=1, ls='--')

    los_patch_lc = jet_obs.L_t_obs[los_coord[0], los_coord[1]] * 4 * np.pi / np.sum(jet.dOmega)
    ax.plot(jet_obs.t_obs[los_coord[0], los_coord[1]], los_patch_lc, color='k', ls='--', lw=1)
    ax.plot(t_gamma, L_gamma, color='r', lw=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(2e-3, 1e5)
    ax.set_ylim(6e34, 2e52)
    ax.set_xlabel(r'$t_{\rm obs}$ [s]')
    ax.set_ylabel('Luminosity [erg/s]')

    cbar = fig.colorbar(sm, ax=ax, location='right', pad=0.02)
    cbar.set_label(r'$\theta$ [deg]')
    vmin, vmax = 0, 90
    y0 = (np.rad2deg(jet.theta_cut) - vmin) / (vmax - vmin)
    y1 = 1.0
    if jet.theta_cut < np.pi/2:
        cbar.ax.add_patch(Rectangle((0, y0), 1, y1 - y0, transform=cbar.ax.transAxes,
                                color='lightgray', clip_on=False))
    
        cbar.ax.text(0.5, (y0 + y1)/2, 'Trapped Zone', ha='center', va='center', transform=cbar.ax.transAxes, rotation=90, fontsize=8)
    cbar.ax.hlines(np.rad2deg(theta_los) + 0.1, 0, 1, color='k', ls='--', lw=1)

    ax.set_title(r'$L_\gamma \, (1 - 10^4 \rm \, keV)$')
    ax.grid(True)
    plt.suptitle(r'$\theta_v = {}^\circ$'.format(int(round(np.rad2deg(theta_los)))), y=0.94)
    plt.tight_layout()
    plt.savefig(path + '/lc_obs_{}.pdf'.format(np.round(np.rad2deg(theta_los))))
    plt.show()
    plt.close()

def plot_jet_spec_obs(jet_obs, theta_los, phi_los, path='./out/'):
    """Plot angle-resolved jet spectra and the summed observer spectrum."""

    jet = jet_obs.outflow
    los_coord = [np.abs(jet.theta[:, 0] - theta_los).argmin(), np.abs(jet.phi[0, :] - phi_los).argmin()]

    n_colors = 256
    cut = np.rad2deg(jet.theta_cut) / 90
    n_cut = int(n_colors * cut)

    cmap_colors = cm.plasma(np.linspace(0, 1, n_cut))
    gray_colors = np.tile(np.array([[0.8, 0.8, 0.8, 1.0]]), (n_colors - n_cut, 1))
    colors = np.vstack([cmap_colors, gray_colors])
    custom_cmap = LinearSegmentedColormap.from_list("plasma+grey", colors)

    norm = mcolors.Normalize(vmin=0, vmax=90)
    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)

    fig, ax = plt.subplots()
    E = jet_obs.radiation.E
    ax.plot(E, E * jet_obs.EN_E, color='k', lw=2)

    stride = max(1, jet.theta.shape[0] // 40)
    for theta_i in range(0, jet.theta.shape[0], stride):
        theta_deg = np.rad2deg(jet.theta[theta_i, 0])
        color = custom_cmap(norm(theta_deg))
        ax.plot(E, E * jet_obs.EN_E_obs[theta_i, 0], color=color, lw=1, ls='--')

    ax.plot(E, E * jet_obs.EN_E_obs[los_coord[0], los_coord[1]], color='k', lw=1, ls='--')

    cbar = fig.colorbar(sm, ax=ax, location='right', pad=0.02)
    cbar.set_label(r'$\theta$ [deg]')
    if jet.theta_cut < np.pi/2:
        y0 = np.rad2deg(jet.theta_cut) / 90
        cbar.ax.add_patch(Rectangle((0, y0), 1, 1 - y0, transform=cbar.ax.transAxes, color='lightgray'))
        cbar.ax.text(0.5, (y0 + 1) / 2, 'Trapped Zone', ha='center', va='center', transform=cbar.ax.transAxes, rotation=90, fontsize=8)
    cbar.ax.hlines(np.rad2deg(theta_los)/90, 0, 1, color='k', ls='--', lw=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.3e3, 1e7])
    ax.set_ylim([1e34, 2e43])
    ax.set_xlabel('E [eV]')
    ax.set_ylabel(r'$E^2 N(E)$ [erg/s]')
    ax.grid()
    plt.title(r'$\theta_v={}^\circ$'.format(int(round(np.rad2deg(theta_los)))))
    plt.tight_layout()
    plt.savefig(path + f'/spec_obs_{np.round(np.rad2deg(theta_los))}.pdf')
    plt.show()
    plt.close()

def plot_E_iso_obs(radiation, opacity, path='./out/'):
    """Plot observed jet isotropic-equivalent energy versus viewing angle."""

    theta_v_list = np.linspace(0, 30, 11)
    theta_rad = np.deg2rad(theta_v_list)

    E_list = []

    for theta_v in theta_rad:
        obs = Observer(radiation, opacity)
        obs.observe(theta_los=theta_v, phi_los=0)
        E_list.append(obs.E_iso_obs)

    plt.figure()

    plt.plot(theta_v_list, E_list, 'k', label=r'$E_{\rm iso}$')

    plt.yscale('log')
    plt.xlabel(r'$\theta_v$ [deg]')
    plt.ylabel(r'$E_{\rm iso}$ [erg]')
    plt.legend()
    plt.grid()

    plt.savefig(path + 'E_iso.pdf')
    plt.show()

# -------------------------------------------------------------------
# Example usage:
# -------------------------------------------------------------------

# define path to save figures
path = './out/'
# make directory if it doesn't exist
os.makedirs(path, exist_ok=True)

# set resolution
n_theta, n_phi = 500, 100
# on-axis isotropic-equivalent energy for given jet core width
E_iso = 1e51
theta_jet = np.deg2rad(5)
# cutoff angle
theta_cut = np.deg2rad(35)
# normalize to on-axis observer
theta_los, phi_los = np.deg2rad(0), np.deg2rad(0)
# model_id = [1: BNS-1, 2: BNS-II, 3: BNS-III/BNS-IV, 4: BH-NS]
model_id = 1

# initialize jet and wind
jet = Jet(
    g0=100,
    E_iso=E_iso,
    eps0=E_iso,
    n_theta=n_theta,
    n_phi=n_phi,
    theta_jet=theta_jet,
    theta_cut=theta_cut,
    jet_struct=1
)

wind = Wind(
    g0=50,
    n_theta=n_theta,
    n_phi=n_phi,
    theta_cut=theta_cut
)

# Define radiation and opacity models
jet_rad = Phenomenological(amati_a=0.41, amati_b=0.83)
opacity = Opacity(M_ej=2e31, v=0.3*c, kappa=1, theta_cut=theta_cut)
wind_rad = SpindownWind(engine=Engine())

jet_src = Radiation(jet, jet_rad).build()
wind_src = Radiation(wind, wind_rad).build()

# Create observer objects and compute observed quantities
jet_obs = Observer(jet_src, opacity)
jet_obs.observe(theta_los=theta_los, phi_los=phi_los)

wind_obs = Observer(wind_src, opacity)
wind_obs.observe(theta_los=theta_los, phi_los=phi_los)

# run an example!
# plot_lc(jet, jet_obs, wind, wind_obs, path=path, model_id=model_id)
# plot_spec(jet, jet_obs, path=path, model_id=model_id)
# plot_E_iso_obs(jet_src, opacity, path=path)
# plot_jet_lc_obs(jet_obs, theta_los, phi_los, path=path)
plot_jet_spec_obs(jet_obs, theta_los, phi_los, path=path)
