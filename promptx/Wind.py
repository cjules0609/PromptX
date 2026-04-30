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

from promptx.Outflow import Outflow

from promptx.helper import *
from promptx.const import *


class Wind(Outflow):
    """Structured magnetar-powered wind defined on a spherical grid."""

    def __init__(self, n_theta=1000, n_phi=100, L=1e48, g0=50, L0=1e48, theta_jet=np.pi/2, theta_cut=np.pi/2, k=0, collapse=False, wind_struct=1):
        """
        Initialize the wind geometry and intrinsic angular structure.
        """

        super().__init__()

        self.theta_jet = theta_jet

        self.k = k
        
        # Store cutoff angle for wind structure
        self.theta_cut = theta_cut

        self.define(g0, L0, wind_struct)

        self.calibrate(L0)

        self.beta = gamma2beta(self.g)
        self.D_on = doppf(self.g, 0)

    def define(self, g0=50, L0=1e48, wind_struct=1, **kwargs):
        """
        Build the intrinsic wind luminosity and Lorentz-factor fields.

        Parameters
        ----------
        g0 : float
            On-axis Lorentz-factor scale.
        L0 : float
            On-axis luminosity per solid angle before calibration.
        wind_struct : int or callable
            Angular structure model.
        """

        self.g0 = g0
        self.dL_dOmega0 = L0
        self.struct = wind_struct

        dL_dOmega_north = eps_grid(
            L0, self.theta, self.phi,
            theta_jet=self.theta_jet,
            struct=self.struct,
            cutoff=self.theta_cut,
            **kwargs
        )

        dL_dOmega_south = eps_grid(
            L0, np.pi - self.theta, self.phi,
            theta_jet=self.theta_jet,
            struct=self.struct,
            cutoff=self.theta_cut,
            **kwargs
        )

        self.dL_dOmega = dL_dOmega_north + dL_dOmega_south

        g_north = gamma_grid(self.g0, self.theta, self.phi, struct=self.struct, theta_jet=self.theta_jet, cutoff=self.theta_cut, **kwargs)
        g_south = gamma_grid(self.g0, np.pi - self.theta, self.phi, struct=self.struct, theta_jet=self.theta_jet, cutoff=self.theta_cut, **kwargs)

        self.g = (g_north + g_south) / 2

    def calibrate(self, L0):
        """
        Rescale the wind so the on-axis observed luminosity matches ``L0``.
        """

        dL_dOmega = self.dL_dOmega.copy()

        # Initial E_iso calculation and normalization (2D)
        L_iso_grid = calc_L_iso_grid(self.theta, self.phi, self.g, dL_dOmega, self.theta_cut, self.dOmega)

        A = L0 / L_iso_grid[0, 0]
        dL_dOmega *= A

        L_iso_grid = calc_L_iso_grid(self.theta, self.phi, self.g, dL_dOmega, self.theta_cut, self.dOmega)

        self.dL_dOmega = dL_dOmega

        self.L_iso_grid = L_iso_grid
        self.L0 = L_iso_grid[0, 0]