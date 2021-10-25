from dataclasses import dataclass
from enum import Enum

import numpy as np
from pyorbital.astronomy import _lmst

from control import ACSState
from env_models import FrameVector
from esoq2 import ESOQ2
from kalman_filter import KalmanFilter, SquareRootSigmaPointKalmanFilter
from model import SatelliteDescriptor
from quaternion import Quaternion
from satellite import Satellite


class EstimationConfig(Enum):
    UKF_FULL_STATE = 0
    UKF_DECOMPOSED = 1
    UKF_KF_ST_ONLY = 2
    UKF_KF_ESQO = 3  # todo: azt ki kellene deríteni, hogy jobb-e, ha az ST q-ját és az ESOQ2 q-ját egyesítjük (kisebb C)


class FrameType(Enum):
    ECI = 0
    ECEF = 1


@dataclass
class LLA(object):
    lon: float
    lat: float
    alt: float  # it is indifferent

    def look_at(self, utc_time=None) -> Quaternion:
        """
        Determines the orientation looking at this LLA
        (i.e. same rotation vector, but + pi angle)
        :param utc_time:
        :return:
        """
        actual_lon = self.lon if utc_time is None else _lmst(utc_time, self.lon)
        q = Quaternion.from_lon_lat(actual_lon, self.lat)
        axis, angle = q.to_axis_angle()

        q_look_at = Quaternion.from_axis_angle(axis, angle + np.pi)

        return q_look_at


class AttitudeTarget(object):

    def __init__(self, frame_type: FrameType, q: Quaternion = None, lla: LLA = None) -> None:
        super().__init__()

        if q is None and lla is None:
            raise ValueError("One of q_init and lla shall be specified, but got None for both")
        elif q is not None and lla is not None:
            raise ValueError("Exactly of q_init and lla shall be specified, but got a not-None value for both")

        self.type = frame_type

        # only one of both will be None
        self.q = q
        self.lla = lla

    def get_current_attitude(self, utc_time=None) -> Quaternion:

        if self.type is FrameType.ECI:
            attitude = self.q
        elif self.type is FrameType.ECEF:
            attitude = self.lla.look_at(utc_time)

        return attitude


class ADS(object):

    def __init__(self, sat: Satellite, f_st, config: EstimationConfig = EstimationConfig.UKF_KF_ESQO,
                 q_init: Quaternion = Quaternion()) -> None:
        super().__init__()

        self.config = config

        if self.config is EstimationConfig.UKF_FULL_STATE:
            self.ukf_full_state_init(q_init, sat)

        elif self.config is EstimationConfig.UKF_DECOMPOSED:
            """Dimensions"""
            # states
            dim_kinematics_states = 6
            dim_dynamics_states = 3

            # observations
            dim_kinematics_obs = 10  # st quaternion + magnetic field and Sun vector
            dim_dynamics_obs = 3

            """Kinematics (UKF)"""
            self.ukf_kin = SquareRootSigmaPointKalmanFilter(sat.kinematics, 0.5e1 * np.eye(dim_kinematics_states),
                                                            0.7e1 * np.eye(dim_kinematics_obs), is_error=True)
            self.ukf_kin.x_pred[0:4] = q_init.q

            """Dynamics (UKF)"""
            self.ukf_dyn = SquareRootSigmaPointKalmanFilter(sat.dynamics, 0.5e-2 * np.eye(dim_dynamics_states),
                                                            4e-1 * np.eye(dim_dynamics_obs))

        elif self.config is EstimationConfig.UKF_KF_ST_ONLY or self.config is EstimationConfig.UKF_KF_ESQO:
            self.ukf_kf_init(q_init, sat)

        else:
            raise NotImplementedError()

    def ukf_kf_init(self, q_init, sat):
        """Dimensions
        """
        # states
        dim_kinematics_states = 7
        dim_dynamics_states = 3
        # observations
        dim_kinematics_obs = 4 if self.config is EstimationConfig.UKF_KF_ST_ONLY else 8  # esoq2 provide one more q
        dim_dynamics_obs = 3
        """Kinematics (KF)"""
        dim_st_obs = 4
        dim_esoq_obs = 4
        self.kf = KalmanFilter(sat.kinematics, 1e-4 * np.eye(dim_kinematics_states), np.diagflat(
            [[1e-3] * dim_st_obs, [1e-1] * dim_esoq_obs]), factor=170)
        self.kf.x_pred[0:4] = q_init.q
        """Dynamics (UKF)"""
        self.ukf = SquareRootSigmaPointKalmanFilter(sat.dynamics, 1e-3 * np.eye(dim_dynamics_states),
                                                    4e-3 * np.eye(dim_dynamics_obs), is_error=False)
        """ESOQ2"""
        if self.config is EstimationConfig.UKF_KF_ESQO:
            self.esoq2 = ESOQ2()

    def ukf_full_state_init(self, q_init, sat):
        """Dimensions"""
        # states
        dim_kinematics_states = 3
        dim_dynamics_states = 3
        # observations
        dim_kinematics_obs = 9  # st quaternion (VECTOR representation) + magnetic field and Sun vector #todo: we also need the quaternion, but +1 (=10) makes a bug in separated chol_update
        dim_dynamics_obs = 3
        """Full state (UKF)"""
        self.ukf_full = SquareRootSigmaPointKalmanFilter(sat.full_model, np.diagflat(
            [[1e-1] * dim_kinematics_states, [1e-2] * dim_dynamics_states]),
                                                         np.diag(np.hstack(
                                                             ([4e-1] * dim_kinematics_obs, [2e-2] * dim_dynamics_obs))),
                                                         is_error=True)
        self.ukf_full.x_pred[0:4] = q_init.q

    def esoq_magnet_sun(self, B: FrameVector, sun: FrameVector):
        b_abc_noisy_unit = B.abc_noisy / np.linalg.norm(B.abc_noisy)
        sun_abc_noisy_unit = sun.abc_noisy / np.linalg.norm(sun.abc_noisy)

        references = np.array([b_abc_noisy_unit, sun_abc_noisy_unit])
        observations = np.array([B.ref / np.linalg.norm(B.ref), sun.ref / np.linalg.norm(sun.ref)])
        self.esoq2.step(observations, references)

        """Experiment
        s = sun.abc / np.linalg.norm(sun.abc)
        x = np.array([1, 1, 1]) # ez lehet akár olyan vektor is, amiben 2 koordináta 0!?
        s_cross = np.cross(x, s)

        S = sun.ref / np.linalg.norm(sun.ref)
        S_cross = np.cross(x, S)

        r = np.array([s, s_cross])
        o = np.array([S, S_cross])

        from copy import deepcopy
        self.esoq2.step(observations, references)
        q_new = deepcopy(self.esoq2.q_opt)

        b_abc_unit = B.abc / np.linalg.norm(B.abc)
        sun_abc_unit = sun.abc / np.linalg.norm(sun.abc)

        references = np.array([b_abc_unit, sun_abc_unit])
        observations = np.array([B.ref / np.linalg.norm(B.ref), sun.ref / np.linalg.norm(sun.ref)])

        self.esoq2.step(observations, references)
        q_old = deepcopy(self.esoq2.q_opt)
        END"""

        return b_abc_noisy_unit @ sun_abc_noisy_unit

    def ukf_kf(self, omega_noisy: np.ndarray, q_st: Quaternion, B: FrameVector, sun: FrameVector, torque: np.ndarray,
               sat: Satellite, acs_state: ACSState, omega_override=None):
        sat.dynamics.x = self.ukf.x_pred  # because the model thinks that the predicted omega is the real one

        """Dynamics (UKF)"""
        f_kwargs = {"inertia": sat.inertia.matrix, "dt": sat.dt, "h_rw": sat.h_rw}
        self.ukf.update(torque, omega_noisy, f_kwargs)

        if omega_override is not None:
            self.ukf.x_pred = omega_override

        if acs_state == ACSState.NOMINAL:
            """Time-variant kinematics"""
            # update the time-variant system variables
            sat.kinematics.A = SatelliteDescriptor.kinematics_with_bias(self.ukf.x_pred, self.kf.x_pred[4:],
                                                                        sat.dt)
            sat.kinematics.x = self.kf.x_pred

            sat.kinematics.x[0:4] = self.q_pred.q

            """ESOQ2"""
            cos_measurement_angle = self.esoq_magnet_sun(B, sun)

            """KF"""
            self.kf.cov_weight = cos_measurement_angle  # update the weight for the covariance
            self.kf.cov_norm_penalty = self.esoq2.q_norm

            self._align_quaternion(q_st)

            kinematics_obs = q_st.q if self.config is EstimationConfig.UKF_KF_ST_ONLY else np.concatenate(
                (q_st.q, self.esoq2.q_opt.q))
            self.kf.update(np.zeros(7), kinematics_obs)

    def _align_quaternion(self, q_st: Quaternion):
        """
        This function checks the predicted attitude quaternion and "aligns" quaternionic measurements
        (by multiplying with -1),
        in case they are "far" from the predicted attitude
        (this is due to the fact that quaternions cover SO(3) twice)
        :param q_st:
        :return: as the arument is mutable, this function has no return value
        """
        if self.q_pred.q @ q_st.q < 0:
            q_st.q = -q_st.q
        if self.q_pred.q @ self.esoq2.q_opt.q < 0:
            self.esoq2.q_opt.q = -self.esoq2.q_opt.q

    def ukf_full_state(self, omega_noisy, q_st: Quaternion, B: FrameVector, sun: FrameVector, torque: np.ndarray,
                       sat: Satellite, omega_overwrite=None):

        """Time-variant kinematics"""
        # update the time-variant system variables
        bias = np.zeros((3,))  # self.ukf_kin.x_pred[3:] # bias
        sat.kinematics.A = SatelliteDescriptor.kinematics_no_bias_state(self.ukf_full.omega_pred, bias, sat.dt)
        sat.kinematics.x[0:4] = self.ukf_full.q_pred.q
        sat.kinematics.x[4:] = bias

        """Dynamics"""
        sat.dynamics.x = self.ukf_full.omega_pred  # because the model thinks that the predicted omega is the real one

        """Full model (UKF)"""
        f_kwargs = {"dynamics": {"inertia": sat.inertia.matrix, "dt": sat.dt, "h_rw": sat.h_rw},
                    "kinematics": {"A": sat.kinematics.A, "B": sat.kinematics.B}}
        h_kwargs = {"kinematics": {"m_igrf": B.ref / np.linalg.norm(B.ref), "s_model": sun.ref}}

        observation = {"kinematics": np.concatenate((q_st.q, B.abc_noisy / np.linalg.norm(B.abc_noisy), sun.abc_noisy)),
                       # todo: itt nem biztos, hogy a vektort kellene elkérni, hanem még forgatni is kellene
                       "dynamics": omega_noisy}  # todo: check coordinate system
        u = {"kinematics": np.zeros(7),
             "dynamics": torque}

        self.ukf_full.update(u, observation, f_kwargs, h_kwargs)

        if omega_overwrite is not None:
            self.ukf_full.omega_pred = omega_overwrite

    def ukf_decomposed(self, omega_noisy: np.ndarray, q_st: Quaternion, B: FrameVector, sun: FrameVector,
                       torque: np.ndarray, sat: Satellite, acs_state: ACSState):
        sat.dynamics.x = self.ukf_dyn.x_pred  # because the model thinks that the predicted omega is the real one

        """Dynamics (UKF)"""
        f_kwargs = {"inertia": sat.inertia.matrix, "dt": sat.dt, "h_rw": sat.h_rw}
        self.ukf_dyn.update(torque, omega_noisy, f_kwargs)

        if acs_state == ACSState.NOMINAL:
            """Time-variant kinematics"""
            # update the time-variant system variables
            sat.kinematics.A = SatelliteDescriptor.kinematics_with_bias(self.ukf_dyn.x_pred,
                                                                        self.ukf_kin.x_pred[3:],
                                                                        sat.dt)
            sat.kinematics.x[0:4] = self.ukf_kin.q_pred.q
            sat.kinematics.x[4:] = self.ukf_kin.x_pred[3:]

            """UKF"""
            kinematics_obs = np.concatenate(
                (q_st.q, B.abc_noisy, sun.abc_noisy))  # todo: check if coordinate system is OK
            f_kwargs = {"A": sat.kinematics.A, "B": sat.kinematics.B}
            # h_kwargs = {"C":sat.kinematics.C}
            h_kwargs = {"m_igrf": B.ref, "s_model": sun.ref}

            self.ukf_kin.update(np.zeros(7), kinematics_obs, f_kwargs, h_kwargs)

    @property
    def q_pred(self):
        if self.config is EstimationConfig.UKF_FULL_STATE:
            q_pred = self.ukf_full.q_pred
        elif self.config is EstimationConfig.UKF_DECOMPOSED:
            q_pred = self.ukf_kin.q_pred
        elif self.config is EstimationConfig.UKF_KF_ST_ONLY or self.config is EstimationConfig.UKF_KF_ESQO:
            q_pred = Quaternion(self.kf.x_pred[0:4])
        else:
            raise NotImplementedError()

        return q_pred

    @property
    def omega_pred(self):
        if self.config is EstimationConfig.UKF_FULL_STATE:
            omega_pred = self.ukf_full.omega_pred
        elif self.config is EstimationConfig.UKF_DECOMPOSED:
            omega_pred = self.ukf_dyn.x_pred
        elif self.config is EstimationConfig.UKF_KF_ST_ONLY or self.config is EstimationConfig.UKF_KF_ESQO:
            omega_pred = self.ukf.x_pred
        else:
            raise NotImplementedError()

        return omega_pred
