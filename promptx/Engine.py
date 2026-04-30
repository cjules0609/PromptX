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
from scipy.integrate import solve_ivp

from .const import G, c


class Engine:
    """
    Proto-magnetar spin-down engine.

    This class evolves the spin frequency and torque-powered luminosities
    without applying any radiation model or observer effects.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================
    def __init__(
        self,
        P0=1e-3,
        B_p=1e15,
        eps=1e-3,
        I=1e45,
        R=1e6,
        t_end=1e10,
        n_steps=1000,
        collapse_time=None,
    ):
        self.P0 = P0
        self.Omega0 = 2 * np.pi / P0

        self.B_p = B_p
        self.eps = eps
        self.I = I
        self.R = R

        self.collapse_time = collapse_time

        # Master time grid (ONLY source of time)
        self.t = np.geomspace(1e-6, t_end, n_steps)

        # Precompute torque coefficients
        self._compute_coefficients()

        # Solve spin evolution
        self.Omega = self._solve_spindown()

    # =========================================================
    # PHYSICS COEFFICIENTS
    # =========================================================
    def _compute_coefficients(self):
        """Precompute electromagnetic and gravitational-wave torque terms."""

        self.a = (32 * G * self.I * self.eps**2) / (5 * c**5)
        self.b = (self.B_p**2 * self.R**6) / (6 * c**3 * self.I)

        # Characteristic timescales (diagnostic only)
        self.t_em = (3 * c**3 * self.I) / (self.B_p**2 * self.R**6 * self.Omega0**2)

        self.t_gw = (5 * c**5) / (
            128 * G * self.I * self.eps**2 * self.Omega0**4
        )

    # =========================================================
    # SPINDOWN EVOLUTION
    # =========================================================
    def _solve_spindown(self):
        """Solve the spin-down ODE on the engine time grid."""

        def ode(t, Omega):
            return -(self.a * Omega**5 + self.b * Omega**3)

        sol = solve_ivp(
            ode,
            (self.t[0], self.t[-1]),
            [self.Omega0],
            t_eval=self.t,
            rtol=1e-6,
            atol=1e-9,
        )

        Omega = sol.y[0]

        # enforce collapse
        if self.collapse_time is not None:
            Omega = np.where(self.t > self.collapse_time, 0.0, Omega)

        return Omega

    # =========================================================
    # PHYSICAL POWER OUTPUTS (NO RADIATION EFFICIENCY)
    # =========================================================
    def em_power(self):
        """Return the electromagnetic spin-down power."""

        return (self.B_p**2 * self.R**6 * self.Omega**4) / (6 * c**3)

    def gw_power(self):
        """Return the gravitational-wave luminosity."""

        return (32 * G * self.I**2 * self.eps**2 * self.Omega**6) / (5 * c**5)

    def total_power(self):
        """Return the total spin-down power."""

        return self.em_power() + self.gw_power()

    # =========================================================
    # INTERPOLATION (for radiation layer)
    # =========================================================
    def omega(self, t):
        """Interpolate the spin frequency onto a new time grid."""

        return np.interp(t, self.t, self.Omega)
