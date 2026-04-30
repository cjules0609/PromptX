# ============================================================================= #
#                  ____  ____   __   _  _  ____  ____  _  _                     #
#                 (  _ \(  _ \ /  \ ( \/ )(  _ \(_  _)( \/ )                    #
#                  ) __/ )   /(  O )/ \/ \ ) __/  )(   )  (                     #
#                 (__)  (__\_) \__/ \_)(_/(__)   (__) (_/\_)                    #
#                                                                               #
# ============================================================================= #
#   PromptX - Prompt X-ray emission modeling of relativistic outflows           #
#   Version 0.3                                                                 #
#   Author: Connery Chen, Yihan Wang, and Bing Zhang                            #
#   License: MIT                                                                #
# ============================================================================= # 

from promptx.Outflow import Outflow

from promptx.helper import *
from promptx.const import *

class Jet(Outflow):
    """Structured relativistic jet defined on a spherical grid."""

    def __init__(self, n_theta=200, n_phi=100, g0=200,
                 E_iso=1e53, eps0=1e53,
                 theta_jet=np.pi/2, theta_cut=np.pi/2,
                 jet_struct=0, **kwargs):

        super().__init__()

        self.theta_jet = theta_jet
        self.theta_cut = theta_cut
        self.E_iso = E_iso

        self.define(g0, eps0, jet_struct, **kwargs)

        self.calibrate(self.E_iso)

        self.beta = gamma2beta(self.g)
        self.D_on = doppf(self.g, 0)

    def define(self, g0, eps0, jet_struct, **kwargs):
        """
        Build the intrinsic jet energy and Lorentz-factor fields.

        Parameters
        ----------
        g0 : float
            On-axis Lorentz-factor scale.
        eps0 : float
            On-axis energy per solid angle before calibration.
        jet_struct : int or callable
            Angular structure model.
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

    def calibrate(self, E_iso):
        """
        Rescale the jet so the on-axis observed energy matches ``E_iso``.
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

    def radiation(self, model=None):

        self.E = np.geomspace(1e2, 1e8, 1000)
        self.t = np.geomspace(1e-3, 1e4, 1000)

        NE_kernel = model.spectrum_kernel(self.E, self.e_iso_grid)
        L_kernel = model.light_curve_kernel(self.t)

        eps_unit = int_spec(self.E, self.E * NE_kernel, E_min=1e3, E_max=10e6)
        mask = eps_unit > 0
        A_spec = np.zeros_like(self.eps)
        A_spec[mask] = self.eps[mask] / eps_unit[mask]

        self.N_E = A_spec[..., np.newaxis] * NE_kernel    

        self.S = int_spec(self.E, self.E * self.N_E, E_min=1e3, E_max=10e6)
        self.S = np.nan_to_num(self.S, nan=0.0)
    
        S_unit = int_lc(self.t, L_kernel)
        A_lc = self.S / S_unit
        self.L = A_lc[..., np.newaxis] * L_kernel

    def refine_grid(self, theta_los, phi_los, n_theta=200, n_phi=100, rotate=False, resample=False):
        """Rotate or downsample the jet grid around a given line of sight."""
        
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
