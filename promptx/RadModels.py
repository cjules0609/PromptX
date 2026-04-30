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

import numpy as np
from scipy.integrate import solve_ivp

from promptx.const import G, c
from promptx.helper import band, fred

class RadiationModel:
    """Abstract interface for source-frame radiation prescriptions."""

    def build_spectrum_kernel(self, E, outflow, **kwargs):
        """Return the spectral kernel on the outflow grid."""
        raise NotImplementedError

    def build_light_curve_kernel(self, t, outflow, **kwargs):
        """Return the temporal kernel on the requested time grid."""
        raise NotImplementedError

    def spectrum_normalization(self, outflow, **kwargs):
        """Return the per-cell spectral normalization field."""
        raise NotImplementedError

    def light_curve_normalization(self, outflow, spectral_energy, **kwargs):
        """Return the per-cell light-curve normalization field."""
        return spectral_energy

    def light_curve_norm_mode(self):
        """Return the light-curve normalization convention."""
        return "integral"

class Phenomenological(RadiationModel):
    """Band-spectrum plus FRED light-curve prompt-emission model."""

    def __init__(self,
                 amati_a=0.41, amati_b=0.83,
                 alpha=-1, beta=-2.3,
                 tau_1=0.1, tau_2=0.35):

        self.amati_a = amati_a
        self.amati_b = amati_b
        self.alpha = alpha
        self.beta = beta

        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def build_spectrum_kernel(self, E, outflow, **kwargs):
        """Build the Band spectral kernel from the Amati relation."""
        e_iso_grid = outflow.e_iso_grid

        Ep = 1e5 * 10**(
            self.amati_a * np.log10(e_iso_grid / 1e51)
            + self.amati_b
        )

        E0 = Ep / (2 + self.alpha)

        NE = band(E, self.alpha, self.beta, E0)

        return NE


    def build_light_curve_kernel(self, t, outflow, **kwargs):
        """Build a FRED temporal kernel."""

        L = fred(t, self.tau_1, self.tau_2)

        return L

    def spectrum_normalization(self, outflow, **kwargs):
        """Normalize spectra by outflow energy per solid angle."""
        return outflow.eps

class SpindownWind(RadiationModel):
    """Spin-down-powered wind radiation model."""

    def __init__(self, engine):
        self.engine = engine

    def default_time_grid(self):
        """Use the engine time grid by default."""
        return self.engine.t.copy()

    def build_spectrum_kernel(self, E, outflow, **kwargs):
        """Return a flat spectral kernel for the wind model."""
        return np.ones(outflow.theta.shape + (E.size,))

    def build_light_curve_kernel(self, t, outflow, **kwargs):
        """Build a light-curve kernel from the engine EM power."""
        power = np.interp(
            t, self.engine.t, self.engine.em_power(), left=0.0, right=0.0
        )
        peak = np.max(power)
        if peak <= 0:
            return np.zeros_like(power)
        return power / peak

    def spectrum_normalization(self, outflow, **kwargs):
        """Normalize spectra by wind luminosity per solid angle."""
        return outflow.dL_dOmega

    def light_curve_normalization(self, outflow, spectral_energy, **kwargs):
        """Normalize light curves by wind luminosity per solid angle."""
        return outflow.dL_dOmega

    def light_curve_norm_mode(self):
        """Normalize wind light curves by their peak value."""
        return "peak"
