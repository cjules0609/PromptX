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

from .const import c

class Opacity:
    """Homologous ejecta opacity model with an angular mask."""

    def __init__(self, M_ej=2e31, v=0.3*c, kappa=1, theta_cut=np.pi/2):
        self.M_ej = M_ej
        self.v = v
        self.kappa = kappa
        self.theta_cut = theta_cut

        self.t_tau = np.sqrt(M_ej * kappa / (4 * np.pi * v**2))

    def tau0(self, t):
        """Return the angle-independent optical-depth scale."""
        return self.kappa * self.M_ej / (4 * np.pi * self.v**2 * t**2)

    def angular_profile(self, theta):
        """Return the angular opacity mask."""
        return np.where(
            np.abs(theta - np.pi/2) < self.theta_cut,
            1.0,
            0.0
        )
    
    def tau(self, theta, t):
        """Return the full angle- and time-dependent optical depth."""

        tau_t = self.tau0(t)[np.newaxis, np.newaxis, :]
        f_theta = self.angular_profile(theta)

        return tau_t * f_theta[:, :, np.newaxis]
