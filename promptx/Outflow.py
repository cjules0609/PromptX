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

from promptx.helper import coord_grid

class Outflow:
    """Base class for angular outflow models defined on a spherical grid."""

    def __init__(self, n_theta=200, n_phi=100):
        self.n_theta = n_theta
        self.n_phi = n_phi

        self.theta_grid = None
        self.phi_grid = None
        self.theta = None
        self.phi = None
        self.dOmega = None

        self.eps = None
        self.g = None
        self.beta = None
        self.D_on = None

        self.e_iso_grid = None

        self.radius = None
        self.t_lab = None
        self.t_comoving = None

        self._build_grid(n_theta, n_phi)

    def emit(self, model, E=None, t=None, **model_kwargs):
        """Build a radiation field for this outflow and radiation model."""
        from promptx.Radiation import Radiation

        return Radiation(self, model, E=E, t=t, **model_kwargs).build()

    def _build_grid(self, n_theta, n_phi):
        """Construct edge-centered and cell-centered spherical grids."""
        theta_bounds = [0, np.pi]
        phi_bounds = [0, 2*np.pi]

        theta_grid, phi_grid = coord_grid(
            n_theta, n_phi,
            theta_bounds,
            phi_bounds
        )

        self.theta_grid = theta_grid
        self.phi_grid = phi_grid

        self.theta = 0.25 * (
            theta_grid[:-1, :-1] + theta_grid[1:, :-1] +
            theta_grid[:-1, 1:] + theta_grid[1:, 1:]
        )

        self.phi = 0.25 * (
            phi_grid[:-1, :-1] + phi_grid[1:, :-1] +
            phi_grid[:-1, 1:] + phi_grid[1:, 1:]
        )

        self.dOmega = (
            (phi_grid[:-1, 1:] - phi_grid[:-1, :-1]) *
            (np.cos(theta_grid[:-1, :-1]) - np.cos(theta_grid[1:, :-1]))
        )
