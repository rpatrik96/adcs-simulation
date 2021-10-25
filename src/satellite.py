from __future__ import annotations

from typing import Optional

import numpy as np

from model import LinearModel, NonLinearModel, ModelWrapper, SatelliteDescriptor
from quaternion import Quaternion
from transforms import HomogeneousTransform, Rotation


class SatellitePart(object):
    def __init__(self, transform: HomogeneousTransform = HomogeneousTransform(),
                 parent: Optional[SatellitePart] = None) -> None:

        # bookkeeping parameters
        self.parent = parent

        # geometric parameters
        self.transform = transform

    def part2abc(self) -> HomogeneousTransform:

        if self.parent is not None:
            return self.transform.transform(self.parent.part2abc())
        else:
            return self.transform

    @property
    def orientation(self) -> Rotation:
        """

        :return: orientation w.r.t. the ABC frame
        """
        return self.part2abc().rotation


class Inertia(object):
    def __init__(self) -> None:
        # todo: min/max for configurations (boom/antenna/both) +interpolation
        self.matrix = np.array([[5.08e-02, 6.54e-5, -1.68e-3],
                                [6.54e-5, 5.62e-2, 4.16e-4],
                                [-1.68e-3, 4.16e-4, 2.19e-2]])

    def disturb(self):

        # off-diagonal elements
        max_off_diag = np.abs(self.matrix - np.diag(self.matrix)).max()
        tri_off_diag = np.tril(np.random.uniform(-.005, .005, size=(3,)), k=-1)
        off_diag_uncertainty = max_off_diag * (tri_off_diag + tri_off_diag.T)

        # diagonal elements
        diag_uncertainty = np.random.uniform(.95, 1.05, size=3)  # factors for +/-15%

        # combine
        uncertainty_matrix = off_diag_uncertainty * (np.ones_like(self.matrix) - np.diag(np.ones(3))) + np.diag(
            diag_uncertainty * np.diag(self.matrix))

        # check positive definiteness
        eigvals, _ = np.linalg.eig(uncertainty_matrix)

        if (eigvals < 0).sum():
            print("Negative eigenvalue found in the disturbed matrix, using the original matrix!")
        else:
            self.matrix = uncertainty_matrix


# needs to be imported here, otherwise the code breaks due to circular dependency
from star_tracker import StarTracker
from actuators import ReactionWheel, Magnetorquer
from control import ActuatorSentinel


class Satellite(object):
    def __init__(self, dt: float, transform: HomogeneousTransform = HomogeneousTransform(),
                 inertia: Inertia = Inertia()) -> None:
        self.dt = dt
        self.inertia = inertia
        self.abc2ned = transform

        self.kinematics = LinearModel(SatelliteDescriptor.kinematics_with_bias(np.zeros(3), np.zeros(3), self.dt),
                                      np.zeros(7),
                                      SatelliteDescriptor.kinematics_obs(
                                          Quaternion.from_sw(0, np.array([2.15e-5, 0., 0.]), False),
                                          Quaternion.from_sw(0, np.array([1., 0., 0.]))
                                      ),
                                      np.array([1, 0, 0, 0, 0, 0, 0]))

        # self.kinematics = SemilinearModel(SatelliteDescriptor.kinematics_with_bias(np.zeros(3), np.zeros(3), self.dt),
        #                                   np.zeros(7),
        #                                   None,
        #                                   np.array([1, 0, 0, 0, 0, 0, 0]))
        self.dynamics = NonLinearModel(SatelliteDescriptor.dynamics,
                                       SatelliteDescriptor.dynamics_obs,
                                       np.array([np.deg2rad(47), np.deg2rad(38), np.deg2rad(62)]))


        # self.full_model = NonLinearModel(
        #     SatelliteDescriptor.state_full,
        #     SatelliteDescriptor.obs_full,
        #     np.array([np.deg2rad(7), np.deg2rad(18), np.deg2rad(12),  # dynamics
        #               1., 0, 0, 0, 0, 0, 0])  # kinematics
        # )

        self.full_model = ModelWrapper(self.kinematics, self.dynamics, np.array([1, 0, 0, 0, 0, 0, 0]))
        # np.zeros(3))

        """Satellite structure"""
        self.root = SatellitePart()
        self.star_tracker = StarTracker(1600, 1200, 90, parent=self.root)

        # RWs
        self.rw_sentinel = ActuatorSentinel()
        self.rw_x = ReactionWheel(dt=self.dt, tau_max=1e-3, h_max=0.012, axis=np.array([1., 0., 0.]), parent=self.root)
        self.rw_y = ReactionWheel(dt=self.dt, tau_max=1e-3, h_max=0.012, axis=np.array([0., 1., 0.]), parent=self.root)
        self.rw_z = ReactionWheel(dt=self.dt, tau_max=1e-3, h_max=0.012, axis=np.array([0., 0., 1.]), parent=self.root)

        # MTQs
        self.mtq_sentinel = ActuatorSentinel()
        self.mtq_x = Magnetorquer(axis=np.array([1., 0., 0.]), m_max=0.36, parent=self.root)
        self.mtq_y = Magnetorquer(axis=np.array([0., 1., 0.]), m_max=0.36, parent=self.root)
        self.mtq_z = Magnetorquer(axis=np.array([0., 0., 1.]), m_max=0.36, parent=self.root)

    def calc_real_m(self, m_ref: np.ndarray) -> np.ndarray:
        """
        Calculates the real actuator dipole moment of the MTQs given a reference

        :param m_ref: MTQ reference torque in ABC frame
        :return:
        """
        m = self.mtq_x.calc_m(m_ref[0]) + self.mtq_y.calc_m(m_ref[1]) + self.mtq_z.calc_m(m_ref[2])

        return m

    def calc_real_tau_rw(self, tau_rw_ref: np.ndarray) -> np.ndarray:
        """
        Calculates the real actuator torque of the RWs given a reference

        :param tau_rw_ref: reference RW torque in ABC frame
        :return:
        """

        tau_rw = self.rw_x.calc_tau(tau_rw_ref[0]) + self.rw_y.calc_tau(tau_rw_ref[1]) + self.rw_z.calc_tau(
            tau_rw_ref[2])

        return tau_rw

    @property
    def h_rw(self):
        return self.rw_x.h_vec + self.rw_y.h_vec + self.rw_z.h_vec

    @h_rw.setter
    def h_rw(self, val: np.ndarray):
        # print(f"val={val}")
        # todo:implement orthogonal projection (axes may not align the xyz-axes of the ABC frame)
        self.rw_x.h = val[0]
        self.rw_y.h = val[1]
        self.rw_z.h = val[2]


if __name__ == "__main__":
    print("\n-----part chaining-----")

    tr = HomogeneousTransform(trans=np.array([1, 0, 0]))
    root = SatellitePart(tr, None)

    tr = HomogeneousTransform(trans=np.array([0, 1, 0]))
    part1 = SatellitePart(tr, root)

    tr = HomogeneousTransform(trans=np.array([1, 1, 0]))
    part2 = SatellitePart(tr, part1)

    tr = HomogeneousTransform(trans=np.array([1, 1, 1]))
    part3 = SatellitePart(tr, part1)

    print(part2.part2abc().matrix)
    print(part3.part2abc().matrix)
