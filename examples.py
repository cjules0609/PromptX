import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle

from helper import *
from main import *

def plot_lc(jet, wind, theta_los, phi_los, path='./out/', model_id=0):
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    fig_lc, ax_lc = plt.subplots()
    ax_lc.plot(jet.t, jet.L_gamma_tot, lw=1, c='r', ls='--', label=r'$10-1000$ keV')
    ax_lc.plot(jet.t, jet.L_X_tot, lw=1, c='g', label=r'$0.3-10$ keV')

    if model_id == 1 or model_id == 2:
        ax_lc.plot(wind.engine.t, wind.L_los, ls='--', lw=1, c='b')
        ax_lc.plot(wind.engine.t, wind.L_dopp, ls='--', lw=1, c='b')
        ax_lc.plot(wind.engine.t, wind.L_X_tot, label=r'$L_{X, \, \rm wind}$', lw=1, c='b')
        if theta_los > wind.theta_cut:
            ax_lc.axvline(wind.engine.t_tau, ls='dotted', c='k', label=r'$t_\tau$', lw=1)
    if model_id == 2:
        ax_lc.axvline(wind.engine.t_coll, ls='-.', c='k', label=r'$t_{\text{coll}}$', lw=1)

    ax_lc.set_xlabel(r'$t$ [s]')
    ax_lc.set_ylabel(r'Luminosity [erg $s^{-1}$ cm$^{-2}$]')
    ax_lc.set_ylim([1e36, 2e52])
    ax_lc.set_xlim([1e-2, 1e8])
    ax_lc.legend(loc='upper right', ncol=2)
    ax_lc.set_title(r'Light Curve')
    ax_lc.set_xscale('log'); ax_lc.set_yscale('log')
    ax_lc.grid()

    plt.savefig(path + 'lc_{}.png'.format(round(np.rad2deg(theta_los), 3)), dpi=300)
    fig_lc.clear()
    plt.close(fig_lc)

def plot_spec(jet, wind, theta_los, phi_los, path='./out/', model_id=0):
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    fig_spec, ax_spec = plt.subplots()

    ax_spec.plot(jet.E, jet.spec_tot * jet.E * jet.E, label=r'$\mathcal{R}_\mathcal{D}$ jet', lw=1, c='r')

    if model_id == 1 or model_id==2:
        ax_spec.plot(wind.E_dopp, wind.A_dopp * wind.N_E_dopp * wind.E_dopp * wind.E_dopp, label=r'$\mathcal{R}_\mathcal{D}$ wind', lw=1, c='b')
    
    ax_spec.set_xlabel(r'$E$ [eV]')
    ax_spec.set_ylabel(r'$E^2 N(E)$ [erg/s]')
    ax_spec.set_ylim([1e42, 1e52])
    ax_spec.set_title(r'Spectrum')
    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')
    ax_spec.grid()

    plt.savefig(path + 'spec_{}.png'.format(round(np.rad2deg(theta_los), 3)), dpi=300)
    fig_spec.clear()
    plt.close(fig_spec)

def plot_jet_lc_obs(jet, theta_los, phi_los, path='./out/'):
    jet.observer(theta_los=theta_los, phi_los=phi_los)


    plt.rcParams.update({'font.size': 12})
    los_coord = nearest_coord(jet.theta, jet.phi, theta_los, phi_los)

    theta_vals = jet.theta[0][::10]
    theta_deg = np.rad2deg(theta_vals)

    norm = mcolors.Normalize(vmin=np.min(theta_deg), vmax=np.max(theta_deg))
    sm = cm.ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])
    colors = cm.rainbow(norm(theta_deg))

    fig, axs = plt.subplots(1, 2)

    for i, theta_i in enumerate(range(0, len(jet.theta[0]), 10)):
        axs[1].plot(jet.t_obs[0, theta_i], jet.L_X_obs[0, theta_i], color=colors[i], lw=1, ls='--')
        axs[0].plot(jet.t_obs[0, theta_i], jet.L_gamma_obs[0, theta_i], color=colors[i], lw=1, ls='--')
        # axs[1].scatter(jet.t_obs[0 , i][np.argmax(jet.L_X_obs[0, i])], np.max(jet.L_X_obs[0, i]), color=colors[i], marker='x', s=20)
        # axs[0].scatter(jet.t_obs[0, i][np.argmax(jet.L_gamma_obs[0, i])], np.max(jet.L_gamma_obs[0, i]), color=colors[i], marker='x', s=20)

    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(2e-3, 1e5)
        ax.set_ylim(6e34, 2e51)
        ax.set_xlabel(r'$t_{\rm obs}$ [s]')
    axs[0].set_ylabel('Luminosity [erg/s]')

    axs[0].plot(jet.t_obs[los_coord[1], los_coord[0]], jet.L_gamma_obs[los_coord[1], los_coord[0]], color='k', ls='--', lw=1)
    axs[0].plot(jet.t, jet.L_gamma_tot, color='r', lw=1)
    # axs[0].axvline(jet.t[np.argmax(jet.L_gamma_tot)], color='r', lw=1)
    # axs[0].axhline(np.max(jet.L_gamma_tot), color='r', lw=1)

    axs[1].plot(jet.t_obs[los_coord[1], los_coord[0]], jet.L_X_obs[los_coord[1], los_coord[0]], color='k', ls='--', lw=1)
    axs[1].plot(jet.t, jet.L_X_tot, color='g', lw=1)
    # axs[1].axvline(jet.t[np.argmax(jet.L_X_tot)], color='g', lw=1)
    # axs[1].axhline(np.max(jet.L_X_tot), color='g', lw=1)

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

    # plt.legend()
    # plt.suptitle(r'Observed Light Curves $\theta={}$'.format(np.round(np.rad2deg(theta_los))))
    plt.tight_layout()
    plt.savefig(path + '/lc_obs_{}.pdf'.format(np.round(np.rad2deg(theta_los))))
    plt.close()

def plot_jet_spec_obs(jet, theta_los, phi_los, path='./out/'):
    jet.observer(theta_los=theta_los, phi_los=phi_los)

    los_coord = nearest_coord(jet.theta, jet.phi, theta_los, phi_los)

    theta_vals = jet.theta[0][::10]
    theta_deg = np.rad2deg(theta_vals)

    norm = mcolors.Normalize(vmin=np.min(theta_deg), vmax=np.max(theta_deg))
    sm = cm.ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])
    colors = cm.rainbow(norm(theta_deg))

    fig, ax = plt.subplots()

    ax.plot(jet.E, jet.E**2 * jet.spec_tot, color='k', lw=2)
    for i, theta_i in enumerate(range(0, len(jet.theta[0]), 10)):
        ax.plot(jet.E, jet.E**2 * jet.N_E[0, theta_i], color=colors[i], lw=1, ls='--')
        ax.plot(jet.E, jet.E**2 * jet.N_E[los_coord[1], los_coord[0]], color='k', ls='--', lw=1)
    ax.plot(jet.E, jet.E**2 * jet.N_E[0, 490], color='r', lw=1)
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

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.3e3 , 1e6)
    ax.set_ylim(6e34, 2e51)
    ax.set_xlabel('E [eV]')
    ax.set_ylabel(r'$E^2 N(E)$ [erg/s]')
    # plt.suptitle(r'Observed Spectrum $\theta={}$'.format(np.round(np.rad2deg(theta_los))))
    plt.tight_layout()
    plt.savefig(path + '/spec_obs_{}.pdf'.format(np.round(np.rad2deg(theta_los))))
    plt.close()

def plot_E_obs(jet, wind, theta_los, phi_los, path='./out/'):
    theta_v_list = np.linspace(0, 90, 31)
    theta_c = np.deg2rad(5)
    theta_cut = 90
    E_iso = 1e51
    n_theta, n_phi = 500, 100

    theta_rad = np.deg2rad(theta_v_list)
    gaussian = E_iso * np.exp(-theta_rad**2 / (2 * theta_c**2))

    S_obs_list = []

    for i, theta_v in enumerate(theta_v_list):
        jet = Jet(g0=300, E_iso=E_iso, eps0=E_iso, n_theta=n_theta, n_phi=n_phi,
                    theta_c=theta_c, theta_cut=theta_cut, struct=1)
        jet.normalize(jet.eps0)
        jet.observer(theta_los=np.deg2rad(theta_v), phi_los=0)
        S_obs_list.append(jet.S_obs)
        print(f"{theta_v:.1f}°: S_obs = {jet.S_obs:.2e}")

    plt.figure()
    plt.plot(theta_v_list, gaussian, color='k', linestyle='--', label='Gaussian profile')
    plt.plot(theta_v_list, S_obs_list, c='k', label=r'$L_\mathrm{obs}$ [erg/s]')

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

path = './out/'
n_theta = 500
n_phi = 100
E_iso_ = 1e51
E_gamma = E_iso_ * (1 - np.cos(5 * np.pi/180))

theta_c = np.deg2rad(5)
theta_cut = np.deg2rad(35)
theta_los = np.deg2rad(0)
phi_los = np.deg2rad(0)

# BNS-I
model_id = 1

E_iso = E_gamma / (1 - np.cos(theta_c))
jet = Jet(g0=200, E_iso=E_iso, eps0=E_iso, n_theta=n_theta, n_phi=n_phi, theta_c=theta_c, theta_cut=theta_cut, struct=1)
jet.normalize(jet.eps0)
jet.observer(theta_los=0, phi_los=0)
jet_eps0 = jet.eps0

jet = Jet(g0=200, eps0=jet_eps0, n_theta=n_theta, n_phi=n_phi, theta_c=theta_c, theta_cut=theta_cut, struct=1)

wind = Wind(g0=50, n_theta=n_theta, n_phi=n_phi, theta_cut=theta_cut)
wind.observer(theta_los=0, phi_los=0)        

plt.rcParams.update({'font.size': 12})

plot_lc(jet, wind, theta_los, phi_los, path=path, model_id=model_id)

plot_spec(jet, wind, theta_los, phi_los, path=path, model_id=model_id)

plot_jet_lc_obs(jet, theta_los, phi_los, path=path)

plot_jet_spec_obs(jet, theta_los, phi_los, path=path)

plot_E_obs(jet, wind, theta_los, phi_los, path=path)