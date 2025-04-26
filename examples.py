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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from helper import *
from main import *

plt.rcParams.update({'font.size': 12})

def plot_lc(jet, wind, theta_los, phi_los, path='./out/', model_id=0):
    """
    Plot the light curves of the jet and wind components.

    Args:
        jet (Jet): Jet object containing the emission data.
        wind (Wind): Wind object containing the emission data.
        theta_los (float): Line-of-sight angle in radians.
        phi_los (float): Azimuthal viewing angle in radians.
        path (str): Directory path to save the figure.
        model_id (int): Model identifier (controls whether wind is plotted).
    """
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    fig_lc, ax_lc = plt.subplots()

    # Plot jet light curves
    ax_lc.plot(jet.t, jet.L_gamma_tot, lw=1, c='r', ls='--', label=r'$10-1000$ keV')
    ax_lc.plot(jet.t, jet.L_X_tot, lw=1, c='g', label=r'$0.3-10$ keV')

    # Plot wind light curves if applicable
    if model_id in [1, 2]:
        ax_lc.plot(wind.engine.t, wind.L_los, ls='--', lw=1, c='b')
        ax_lc.plot(wind.engine.t, wind.L_dopp, ls='--', lw=1, c='b')
        ax_lc.plot(wind.engine.t, wind.L_X_tot, lw=1, c='b', label=r'$L_{X,\, \rm wind}$')

        if theta_los > wind.theta_cut:
            ax_lc.axvline(wind.engine.t_tau, ls='dotted', c='k', label=r'$t_\tau$', lw=1)

    if model_id == 2:
        ax_lc.axvline(wind.engine.t_coll, ls='-.', c='k', label=r'$t_{\text{coll}}$', lw=1)

    ax_lc.set_xlabel(r'$t$ [s]')
    ax_lc.set_ylabel(r'Luminosity [erg $s^{-1}$ cm$^{-2}$]')
    ax_lc.set_xlim([1e-2, 1e8])
    ax_lc.set_ylim([1e36, 2e52])
    ax_lc.set_xscale('log')
    ax_lc.set_yscale('log')
    ax_lc.grid()
    ax_lc.legend(loc='upper right', ncol=2)

    plt.savefig(path + f'lc_{round(np.rad2deg(theta_los), 3)}.png', dpi=300)
    plt.close(fig_lc)

def plot_spec(jet, wind, theta_los, phi_los, path='./out/', model_id=0):
    """
    Plot the spectra of the jet and wind components.

    Args:
        (same as plot_lc)
    """
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    fig_spec, ax_spec = plt.subplots()

    ax_spec.plot(jet.E, jet.spec_tot * jet.E**2, label=r'$\mathcal{R}_\mathcal{D}$ jet', lw=1, c='r')

    if model_id in [1, 2]:
        ax_spec.plot(wind.E_dopp, wind.A_dopp * wind.N_E_dopp * wind.E_dopp**2, label=r'$\mathcal{R}_\mathcal{D}$ wind', lw=1, c='b')

    ax_spec.set_xlabel(r'$E$ [eV]')
    ax_spec.set_ylabel(r'$E^2 N(E)$ [erg/s]')
    ax_spec.set_xlim([0.3e3, 1e6])
    ax_spec.set_ylim([1e42, 1e52])
    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')
    ax_spec.grid()
    ax_spec.legend()

    plt.savefig(path + f'spec_{round(np.rad2deg(theta_los), 3)}.png', dpi=300)
    plt.close(fig_spec)

def plot_jet_lc_obs(jet, theta_los, phi_los, path='./out/'):
    """
    Plot light curves of different emission regions and the total integrated light curve.

    Args:
        (same as plot_lc)
    """
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    plt.rcParams.update({'font.size': 12})
    los_coord = nearest_coord(jet.theta, jet.phi, theta_los, phi_los)

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

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))

    for theta_i in range(0, len(jet.theta[0]), 20):
        theta_deg = np.rad2deg(jet.theta[0])[theta_i]
        color = custom_cmap(norm(theta_deg))
        axs[1].plot(jet.t_obs[0, theta_i], jet.L_X_obs[0, theta_i], color=color, lw=1, ls='--')
        axs[0].plot(jet.t_obs[0, theta_i], jet.L_gamma_obs[0, theta_i], color=color, lw=1, ls='--')

    axs[0].plot(jet.t_obs[los_coord[1], los_coord[0]], jet.L_gamma_obs[los_coord[1], los_coord[0]], color='k', ls='--', lw=1)
    axs[0].plot(jet.t, jet.L_gamma_tot, color='r', lw=1)

    axs[1].plot(jet.t_obs[los_coord[1], los_coord[0]], jet.L_X_obs[los_coord[1], los_coord[0]], color='k', ls='--', lw=1)
    axs[1].plot(jet.t, jet.L_X_tot, color='g', lw=1)

    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(2e-3, 1e5)
        ax.set_ylim(6e34, 2e52)
        ax.set_xlabel(r'$t_{\rm obs}$ [s]')
    axs[0].set_ylabel('Luminosity [erg/s]')

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

    axs[0].set_title(r'$L_\gamma \, (10 - 1000 \rm \, keV)$')
    axs[1].set_title(r'$L_X \, (0.3 - 10 \rm \, keV)$')
    axs[0].grid(True)
    axs[1].grid(True)
    plt.suptitle(r'$\theta_v = {}^\circ$'.format(int(round(np.rad2deg(theta_los)))), y=0.94)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(path + '/lc_obs_{}.pdf'.format(np.round(np.rad2deg(theta_los))))
    plt.close()

def plot_jet_spec_obs(jet, theta_los, phi_los, path='./out/'):
    """
    Plot spectra of different emission regions and the total integrated spectrum.

    Args:
        (same as plot_lc)
    """
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    los_coord = nearest_coord(jet.theta, jet.phi, theta_los, phi_los)

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
    ax.plot(jet.E, jet.E**2 * jet.spec_tot, color='k', lw=2)

    for theta_i in range(0, len(jet.theta[0]), 20):
        theta_deg = np.rad2deg(jet.theta[0])[theta_i]
        color = custom_cmap(norm(theta_deg))
        ax.plot(jet.E, jet.E**2 * jet.N_E_obs[0, theta_i], color=color, lw=1, ls='--')

    ax.plot(jet.E, jet.E**2 * jet.N_E_obs[los_coord[1], los_coord[0]], color='k', lw=1, ls='--')

    cbar = fig.colorbar(sm, ax=ax, location='right', pad=0.02)
    cbar.set_label(r'$\theta$ [deg]')
    if jet.theta_cut < np.pi/2:
        y0 = np.rad2deg(jet.theta_cut) / 90
        cbar.ax.add_patch(Rectangle((0, y0), 1, 1 - y0, transform=cbar.ax.transAxes, color='lightgray'))
        cbar.ax.text(0.5, (y0 + 1) / 2, 'Trapped Zone', ha='center', va='center', transform=cbar.ax.transAxes, rotation=90, fontsize=8)
    cbar.ax.hlines(np.rad2deg(theta_los)/90, 0, 1, color='k', ls='--', lw=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.3e3, 1e6)
    ax.set_ylim(6e34, 2e51)
    ax.set_xlabel('E [eV]')
    ax.set_ylabel(r'$E^2 N(E)$ [erg/s]')
    ax.grid()
    plt.title(r'$\theta_v={}^\circ$'.format(int(round(np.rad2deg(theta_los)))))
    plt.tight_layout()
    plt.savefig(path + f'/spec_obs_{np.round(np.rad2deg(theta_los))}.pdf')
    plt.close()

def plot_E_iso_obs(jet, path='./out/'):
    """
    Plot observed isotropic-equivalent energy of the jet vs viewing angle.

    Args:
        (same as plot_lc)
    """
    theta_v_list = np.linspace(0, 90, 31)
    theta_c = np.deg2rad(5)
    E_iso = 1e51
    theta_rad = np.deg2rad(theta_v_list)
    gaussian = E_iso * np.exp(-theta_rad**2 / (2 * theta_c**2))

    S_obs_list = []
    for theta_v in theta_v_list:
        jet.observer(theta_los=np.deg2rad(theta_v), phi_los=0)
        S_obs_list.append(jet.E_iso_obs)

    plt.figure()
    plt.plot(theta_v_list, gaussian, 'k--', label='Gaussian profile')
    plt.plot(theta_v_list, S_obs_list, 'k', label=r'$L_\mathrm{obs}$')

    plt.yscale('log')
    plt.xlim([0, 90])
    plt.ylim([1e40, 1e53])
    plt.xlabel(r'$\theta_\mathrm{v}$ [deg]')
    plt.ylabel('Luminosity [erg/s]')
    plt.title('Observed Fluence vs Viewing Angle')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + '/E_obs.pdf')
    plt.close()

# -------------------------------------------------------------------
# Example usage:
# -------------------------------------------------------------------

# define path to save figures
path = './out/'
# set resolution
n_theta, n_phi = 500, 100
# on-axis isotropic-equivalent energy for given jet core width
E_iso = 1e51
theta_c = np.deg2rad(5)
# cutoff angle
theta_cut = np.deg2rad(35)
# normalize to on-axis observer
theta_los, phi_los = np.deg2rad(0), np.deg2rad(0)
# model_id = [1: BNS-1, 2: BNS-II, 3: BNS-III/BNS-IV, 4: BH-NS]
model_id = 1

# initialize jet and wind
jet = Jet(g0=200, E_iso=E_iso, eps0=E_iso, n_theta=n_theta, n_phi=n_phi, theta_c=theta_c, theta_cut=theta_cut, struct=1)
wind = Wind(g0=50, n_theta=n_theta, n_phi=n_phi, theta_cut=theta_cut)
wind.observer(theta_los=0, phi_los=0)

# run an example!
# plot_lc(jet, wind, theta_los, phi_los, path=path, model_id=model_id)
# plot_spec(jet, wind, theta_los, phi_los, path=path, model_id=model_id)
plot_jet_lc_obs(jet, theta_los, phi_los, path=path)
# plot_jet_spec_obs(jet, theta_los, phi_los, path=path)
# plot_E_iso_obs(jet, path=path)