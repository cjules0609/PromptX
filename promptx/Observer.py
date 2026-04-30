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
from promptx.helper import (
    nearest_coord,
    angular_d,
    doppf,
    interp_lc,
    int_spec,
    int_lc
)

class Observer:
    """Project source-frame radiation into the observer frame."""

    def __init__(self, radiation, opacity):
        self.radiation = radiation
        self.outflow = radiation.outflow
        self.opacity = opacity

    # =========================================================
    # GEOMETRY
    # =========================================================
    def _geometry(self, theta_los, phi_los):
        """Compute observer-dependent angular separations on the outflow grid."""

        outflow = self.outflow

        self.los_coord = nearest_coord(
            outflow.theta, outflow.phi, theta_los, phi_los
        )

        self.theta_obs = angular_d(
            outflow.theta[self.los_coord[0], 0],
            outflow.theta,
            outflow.phi[0, self.los_coord[1]],
            outflow.phi
        )

    # =========================================================
    # DOPPLER FACTOR
    # =========================================================
    def _doppler(self):
        """Compute the Doppler-factor ratio for each emitting cell."""

        outflow = self.outflow

        D_off = doppf(outflow.g, self.theta_obs)
        self.R_D = D_off / outflow.D_on

    # =========================================================
    # TIME MAPPING (geometry-induced arrival times)
    # =========================================================
    def _time_transform(self):
        """Map source-frame times to observer-frame arrival times."""

        outflow = self.outflow
        radiation = self.radiation

        self.t_obs = radiation.t * (
            1 - outflow.beta[..., np.newaxis] * np.cos(self.theta_obs)[..., np.newaxis]
        ) / (1 - outflow.beta[..., np.newaxis])

    # =========================================================
    # PROJECT EMISSION (KEEP ENERGY UNTIL LATER IF NEEDED)
    # =========================================================
    def _project_emission(self):
        """Apply Doppler and opacity weights to the emitted radiation."""

        outflow = self.outflow
        radiation = self.radiation

        # spectral emissivity per solid angle
        self.EN_E_obs = (
            radiation.E * radiation.N_E * self.R_D[..., np.newaxis]**3
        )
        self.EN_E_obs *= outflow.dOmega[..., np.newaxis]

        # bolometric luminosity per solid angle
        self.L_t_obs = (
            radiation.L_t * self.R_D[..., np.newaxis]**4
        )

        tau = self.opacity.tau(self.theta_obs, radiation.t)
        T = np.exp(-tau)

        self.EN_E_obs *= T
        self.L_t_obs *= T

        self.L_t_obs *= outflow.dOmega[..., np.newaxis]

        return self.L_t_obs, self.EN_E_obs

    # =========================================================
    # BOLOMETRIC COLLAPSE (safe because no energy dependence needed)
    # =========================================================
    def _integrate_bolometric(self):
        """Collapse the angle-resolved emission into observer-frame totals."""

        outflow = self.outflow

        self.EN_E = np.sum(self.EN_E_obs, axis=(0, 1))

        self.t, self.L_t = interp_lc(self.t_obs, self.L_t_obs)

        weight = np.sum(outflow.dOmega[..., np.newaxis], axis=(0, 1))

        self.EN_E /= weight
        self.L_t /= weight

    def _isotropic_equivalent(self):
        """Convert observer-frame spectra and light curves to isotropic-equivalent quantities."""
        outflow = self.outflow
        radiation = self.radiation

        self.eps_bar = int_spec(
            radiation.E,
            self.EN_E,
            E_min=1e3,
            E_max=10e6
        )

        self.E_iso_obs = 4 * np.pi * self.eps_bar

        self.L_t *= 4 * np.pi
        self.L_iso_obs = int_lc(self.t, self.L_t)

    # =========================================================
    # PUBLIC API
    # =========================================================
    def observe(self, theta_los=0.0, phi_los=0.0):
        """Run the full observer-frame projection for one line of sight."""

        self._geometry(theta_los, phi_los)
        self._doppler()
        self._time_transform()
        self._project_emission()
        self._integrate_bolometric()
        self._isotropic_equivalent()

    def spectrum(self):
        """Return the observer-frame time-averaged spectrum."""
        return self.radiation.E, self.EN_E

    def light_curve(self, E_min=None, E_max=None):
        """Return the bolometric or band-limited observer-frame light curve."""

        if E_min is None and E_max is None:
            return self.t, self.L_t
        
        outflow = self.outflow
        radiation = self.radiation

        mask = outflow.eps > 0

        frac = np.zeros_like(outflow.eps)
        frac[mask] = int_spec(
            radiation.E,
            radiation.E * radiation.N_E,
            E_min=E_min,
            E_max=E_max,
        )[mask] / outflow.eps[mask]

        L_band = self.L_t_obs * frac[..., None]

        t, L_band_t = interp_lc(self.t_obs, L_band)

        weight = np.sum(outflow.dOmega)
        L_band_t /= weight

        L_band_t *= 4 * np.pi

        return t, L_band_t
