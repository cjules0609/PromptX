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

from promptx.helper import int_lc, int_spec


class Radiation:
    """Build source-frame spectra and light curves for an outflow."""

    def __init__(self, outflow, model, E=None, t=None, **model_kwargs):
        self.outflow = outflow
        self.model = model
        self.model_kwargs = model_kwargs

        self.E = E
        self.t = t

        self.N_E = None
        self.L_t = None
        self.S = None

    def _default_energy_grid(self):
        """Return the default energy grid."""
        return np.geomspace(1e2, 1e8, 1000)

    def _default_time_grid(self):
        """Return the default time grid."""
        return np.geomspace(1e-3, 1e4, 1000)

    def _resolve_time_grid(self):
        """Choose the explicit, model-provided, or default time grid."""
        if self.t is not None:
            return self.t

        if hasattr(self.model, "default_time_grid"):
            t_model = self.model.default_time_grid()
            if t_model is not None:
                return t_model

        return self._default_time_grid()

    def build(self):
        """Assemble normalized source-frame spectra and light curves."""
        outflow = self.outflow

        if self.E is None:
            self.E = self._default_energy_grid()
        self.t = self._resolve_time_grid()

        NE_kernel = self.model.build_spectrum_kernel(
            self.E, outflow, **self.model_kwargs
        )
        L_kernel = self.model.build_light_curve_kernel(
            self.t, outflow, **self.model_kwargs
        )
        spec_norm = self.model.spectrum_normalization(
            outflow, **self.model_kwargs
        )

        eps_unit = int_spec(self.E, self.E * NE_kernel, E_min=1e3, E_max=10e6)
        mask = eps_unit > 0
        A_spec = np.zeros_like(spec_norm)
        A_spec[mask] = spec_norm[mask] / eps_unit[mask]

        self.N_E = A_spec[..., np.newaxis] * NE_kernel

        self.S = int_spec(self.E, self.E * self.N_E, E_min=1e3, E_max=10e6)
        self.S = np.nan_to_num(self.S, nan=0.0)

        lc_mode = self.model.light_curve_norm_mode()
        if lc_mode == "integral":
            L_unit = int_lc(self.t, L_kernel)
        elif lc_mode == "peak":
            L_unit = np.max(L_kernel)
        else:
            raise ValueError(f"Unsupported light-curve normalization mode: {lc_mode}")

        if L_unit <= 0:
            A_lc = np.zeros_like(self.S)
        else:
            lc_norm = self.model.light_curve_normalization(
                outflow, self.S, **self.model_kwargs
            )
            A_lc = lc_norm / L_unit
        self.L_t = A_lc[..., np.newaxis] * L_kernel

        return self
