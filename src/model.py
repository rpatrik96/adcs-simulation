from __future__ import annotations

import numpy as np

from alg_utils import cross_prod_mat
from quaternion import Quaternion


def quat_state_transition(x, u, A, B):
    x = A @ x + B @ u

    # print(np.linalg.norm(self.x[:4]))
    # if (x[:4].sum() <1e-9):
    #     x[0]=1

    x[:4] /= np.linalg.norm(
        x[:4])  # todo: check for better solution (e.g. the own idea of correcting the w error)
    # print(f"q_norm={np.linalg.norm(self.x[:4])}")
    return x


def lin_observation(x, C):
    return C @ x


class Model(object):

    def __init__(self, f: callable, h: callable, x0: np.ndarray) -> None:
        super().__init__()
        self.f = f
        self.h = h
        self.x = x0

    def predict(self, u: np.ndarray, *args):
        raise NotImplementedError

    def observe(self, *args):
        raise NotImplementedError


class LinearModel(Model):

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, x0: np.ndarray) -> None:
        super().__init__(quat_state_transition, lin_observation, x0)

        self.A = A
        self.B = B
        self.C = C

    def predict(self, u: np.ndarray, *args) -> np.ndarray:
        return self.f(self.x, u, self.A, self.B)

    def observe(self, *args):
        return self.h(self.x, self.C)


# class SemilinearModel(Model):
#
#     def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, x0: np.ndarray) -> None:
#         super().__init__(quat_state_transition, SatelliteDescriptor.kinematic_obs_full, x0)
#
#         self.A = A
#         self.B = B
#         self.C = C
#
#     def predict(self, u: np.ndarray, *args) -> np.ndarray:
#         return self.f(self.x, u, self.A, self.B)
#
#     def observe(self, *args):
#         return self.h(*args)


class NonLinearModel(Model):

    def __init__(self, f: callable, h: callable, x0: np.ndarray) -> None:
        super().__init__(f, h, x0)

    def predict(self, u: np.ndarray, *args) -> np.ndarray:
        # self.x = self.f(self.x, u, *args)
        return self.f(self.x, u, *args)

    def observe(self, *args) -> None:
        return self.h(self.x, *args)


class ModelWrapper(Model):

    def __init__(self, kinematics: Model, dynamics: Model, x0: np.ndarray) -> None:
        self.kinematics = kinematics
        self.dynamics = dynamics

        def f_wrapper(x, **kwargs):
            return np.hstack((
                self.kinematics.f(x[:4], kwargs["u"]["kinematics"], **kwargs["kinematics"]),
                self.dynamics.f(x[4:], kwargs["u"]["dynamics"], **kwargs["dynamics"])
            ))

        def h_wrapper(x, **kwargs):
            q_pred = Quaternion(x[:4])
            return np.hstack((
                q_pred.q,
                q_pred.rotate_vector(kwargs["kinematics"]["m_igrf"]),
                q_pred.rotate_vector(kwargs["kinematics"]["s_model"]),
                self.dynamics.h(x[4:])
            ))


        super().__init__(f_wrapper, h_wrapper, x0)

    def predict(self, u: np.ndarray, *args):
        pass

    def observe(self, *args):
        pass


class SatelliteDescriptor(object):
    @staticmethod
    def dynamics(omega: np.ndarray, u: np.ndarray, inertia: np.ndarray, dt: float, h_rw: np.ndarray) -> np.ndarray:
        # print(f"sat={inertia@omega}, h_rw={h_rw}")
        return omega + np.linalg.inv(inertia) @ (u - cross_prod_mat(omega) @ (inertia @ omega + h_rw)) * dt

    @staticmethod
    def kinematics_with_bias(w: np.ndarray, bias: np.ndarray, dt: float) -> np.ndarray:
        state_dim = 4 + w.shape[0]
        state_transition = np.eye(state_dim)
        state_transition[0:4, 0:4] = np.eye(4) + .5 * Quaternion.from_sw(0, w - bias,
                                                                         False).quatprod_matrix * dt  # todo: multiplicative quaternion kinematics?! (no normalization needed basically, as quat_rot preserves the norm)

        return state_transition

    @staticmethod
    def kinematics_no_bias_state(w: np.ndarray, bias: np.ndarray, dt: float) -> np.ndarray:
        state_transition = np.eye(4) + .5 * Quaternion.from_sw(0, w - bias,
                                                               False).quatprod_matrix * dt  # todo: multiplicative quaternion kinematics?! (no normalization needed basically, as quat_rot preserves the norm)

        return state_transition

    @staticmethod
    def dynamics_obs(omega: np.ndarray) -> np.ndarray:
        return np.eye(omega.shape[0]) @ omega

    @staticmethod
    def kinematics_obs(m_igrf: Quaternion, s_model: Quaternion) -> np.ndarray:
        """

        :param m_igrf: magnetic field vector from the IGRF model rotated with the quaternion state from the left
        :param s_model: Sun vector from the Sun model rotated with the quaternion state from the left
        :return:
        """
        obs_mat = np.zeros((12, 7))
        obs_mat[0:4, 0:4] = np.eye(4)
        obs_mat[4:8, 0:4] = m_igrf.quatprod_matrix
        obs_mat[8:12, 0:4] = s_model.quatprod_matrix

        return obs_mat

    @staticmethod
    def kinematics_obs_simple():
        """
        Observation model for attitude kinematics with one quaternion measurements

        :return: observation matrix (C)
        """
        C = np.zeros((4, 7))
        C[0:4, 0:4] = np.eye(4)
        return C

    @staticmethod
    def kinematics_obs_double():
        """
        Observation model for attitude kinematics with one quaternion measurements

        :return: observation matrix (C)
        """
        C = np.zeros((8, 7))
        C[0:4, 0:4] = np.eye(4)
        C[4:8, 0:4] = np.eye(4)

        return C

    @staticmethod
    def kinematic_obs_full(kinematics_state: np.ndarray, m_igrf: np.ndarray, s_model: np.ndarray) -> np.ndarray:
        """
        Observation model for attitude kinematics (nonlinear form), two vector and one quaternion measurement.
        Returns the transformed measurement vectors concatenated

        :param kinematics_state: quaternion estimate + bias estimate
        :param m_igrf: magnetic field vector
        :param s_model: sun vector
        :return: np.ndarray containing all measurements ordered as (q.q, m_rotated, s_rotated)
        """

        q = Quaternion(kinematics_state[:4], True)
        m_rotated = q.rotate_vector(m_igrf)
        s_rotated = q.rotate_vector(s_model)

        return np.concatenate((q.q, m_rotated, s_rotated))

    @classmethod
    def obs_full(cls, omega: np.ndarray, q: Quaternion, m_igrf: np.ndarray, s_model: np.ndarray) -> np.ndarray:
        """
        Wrapper for the dynamic and kinematic observations

        :param omega: angular velocity vector
        :param q: quaternion estimate
        :param m_igrf: magnetic field vector
        :param s_model: sun vector
        :return: np.ndarray containing all observations ordered as (obs_dynamics, obs_kinematics)
        """
        obs_dynamics = cls.dynamics_obs(omega)
        obs_kinematics = cls.kinematic_obs_full(q.q, m_igrf, s_model)

        return np.concatenate((obs_dynamics, obs_kinematics))

    @classmethod
    def state_full(cls, omega: np.ndarray, tau: np.ndarray, inertia: np.ndarray, dt: float, h_rw: np.ndarray,
                   bias: np.ndarray) -> np.ndarray:
        """
        Wrapper for the dynamic and kinematic state equations

        :param omega: angular velocity vector
        :param tau: torque vector affecting the model (disturbance estimate can be included if present)
        :param inertia: inertia matrix of the model
        :param dt: time step of the simulation
        :param h_rw: intrinsic angular momentum (e.g. of reaction wheels)
        :param bias: sensor bias vector
        :return: np.ndarray containing all states ordered as (x_dynamics, x_kinematics)
        """
        x_dynamics = cls.dynamics(omega, tau, inertia, dt, h_rw)
        x_kinematics = cls.kinematics_with_bias(omega, bias, dt)
        x_kinematics[0:4] /= np.linalg.norm(x_kinematics[0:4])  # todo:check if needed here

        return np.concatenate((x_dynamics, x_kinematics))
