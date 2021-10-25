from typing import Optional

import numpy as np

from satellite import SatellitePart
from transforms import HomogeneousTransform


class Actuator(SatellitePart):
    # todo: the slope of the torque should be constrained
    def __init__(self, axis: np.ndarray, transform: HomogeneousTransform = HomogeneousTransform(),
                 parent: Optional[SatellitePart] = None) -> None:
        super().__init__(transform, parent)

        self.axis = axis


class Magnetorquer(Actuator):
    # todo: add permeability if needed

    def __init__(self, axis: np.ndarray, m_max: float, transform: HomogeneousTransform = HomogeneousTransform(),
                 parent: Optional[SatellitePart] = None) -> None:
        super().__init__(axis, transform, parent)
        self.m_max = m_max

    def calc_m(self, m_ref: float) -> np.ndarray:
        m = np.clip(m_ref, -self.m_max, self.m_max)
        return m* self.part2abc().rotation.vec_mult(self.axis)


class ReactionWheel(Actuator):

    def __init__(self, axis: np.ndarray, tau_max: float, h_max: float, dt: float,
                 transform: HomogeneousTransform = HomogeneousTransform(),
                 parent: Optional[SatellitePart] = None) -> None:
        super().__init__(axis, transform, parent)
        self.dt = dt
        self.tau_max = tau_max
        self.h_max = h_max
        self._h = 0  # angular momentum

    def calc_tau(self, tau_ref) -> np.ndarray:

        tau = np.clip(tau_ref, -self.tau_max, self.tau_max)

        # reset torque if wheel would go into saturation
        # this approach avoids numeric differentiation, but
        # allows an error up to tau * self.dt,
        # which should be negligible (tau*st ~1e-3*.5, h ~1e-2)
        # if np.abs(self.h + tau * self.dt) > self.h_max:
        #     tau = 0.
        # else:
        #     pass
        #     # self.h += tau * self.dt

        return tau * self.part2abc().rotation.vec_mult(self.axis)

    @property
    def h_vec(self):
        return self.h * self.part2abc().rotation.vec_mult(self.axis)


    @property
    def h(self):
        return self._h


    @h.setter
    def h(self, value):
        self._h = np.clip(value, -self.h_max, self.h_max)