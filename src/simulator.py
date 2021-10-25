from copy import deepcopy

import numpy as np

from control import ACS
from control import ACSState
from env_models import EnvModelWrapper, FrameVector
from estimation import AttitudeTarget, FrameType, LLA
from estimation import EstimationConfig, ADS
from model import SatelliteDescriptor
from quaternion import Quaternion
from satellite import Satellite

NOISE_FLAG = 1
INCLUDE_DIST_FLAG=1

ESTIMATION = EstimationConfig.UKF_KF_ESQO


class Simulator(object):
    def __init__(self, q_target=Quaternion(np.array([5, 1, 2, 3])), timestep: int = 250, t_sat_sample: int = 250,
                 h_ref: float = 0.007, f_st: int = 4, omega_norm_deg=80, uncertain_inertia=False, nominal_only=False,
                 detumbling_disturb=False, use_eci=False, use_ecef=False, fixed_init=True, tumble_off=False):
        pass
        """Components"""
        self.env = EnvModelWrapper(timestep=timestep)
        self.sat_ideal = Satellite(dt=self.env.timestep.total_seconds())

        if ESTIMATION is EstimationConfig.UKF_FULL_STATE:
            pass
        elif ESTIMATION is EstimationConfig.UKF_DECOMPOSED:
            pass
        elif ESTIMATION is EstimationConfig.UKF_KF_ST_ONLY:
            self.sat_ideal.kinematics.C = SatelliteDescriptor.kinematics_obs_simple()
        elif ESTIMATION is EstimationConfig.UKF_KF_ESQO:
            self.sat_ideal.kinematics.C = SatelliteDescriptor.kinematics_obs_double()
        else:
            raise NotImplementedError()


        if tumble_off is True:
            self.sat_ideal.dynamics.x = np.array([np.deg2rad(3), np.deg2rad(4), np.deg2rad(5)])


        if detumbling_disturb or uncertain_inertia:
            rand_vec_deg = np.random.uniform(-30, 30, 3)
            rand_vec_deg = rand_vec_deg / np.linalg.norm(rand_vec_deg) * omega_norm_deg

            self.sat_ideal.dynamics.x = np.deg2rad(rand_vec_deg)

        self.sat = deepcopy(self.sat_ideal)
        self.ukf_sat = deepcopy(self.sat_ideal)
        self.sat.dt = t_sat_sample / 1000.0
        self.f_st = f_st

        self.dt_ratios = int(self.sat.dt / self.sat_ideal.dt)

        if uncertain_inertia:
            self.sat.inertia.disturb()

        """ADS"""
        self.q_true = Quaternion()
        self.q_st = Quaternion()

        q_init = Quaternion()
        self.ads = ADS(self.sat, f_st, ESTIMATION, q_init)

        self.ads.ukf_full_state_init(q_init, self.ukf_sat)

        """ACS"""
        omega_target = np.zeros(3)

        if fixed_init is True:
            if use_eci and use_ecef:
                raise ValueError("Both ECI and ECEF options are selected, use only one of them!")

            elif use_eci is False and use_ecef is False:
                raise ValueError("None of ECI and ECEF options are selected, use one of them!")

            if use_eci is True:
                q_target = Quaternion(np.array([1, -1, 2, 3]))
                self.attitude_target = AttitudeTarget(FrameType.ECI, q=q_target)

            elif use_ecef is True:
                budapest_lla = LLA(lon=np.deg2rad(19.040236), lat=np.deg2rad(47.497913), alt=0)
                self.attitude_target = AttitudeTarget(FrameType.ECEF, lla=budapest_lla)

        else:
            if use_eci:
                self.attitude_target = AttitudeTarget(FrameType.ECI, q=q_target)
            elif use_ecef:
                raise NotImplementedError("ECEF frame is not supported for arbitrary quaternions!")

        self.h_ref = h_ref
        self.acs = ACS(self.attitude_target.get_current_attitude(self.env.time), omega_target, self.sat.dt,
                       self.h_ref * np.ones(3), self.sat.calc_real_m, self.sat.calc_real_tau_rw, hybrid_c=8e-4,
                       dump_gain=4e-3,  # todo: hybrid_c=3e-4 is ok
                       nominal_only=nominal_only)

        """Miscellaneous"""
        self.i = 0

    def step(self):
        """
        Main simulator loop
        :return:
        """
        self.env_step()
        self.acs_step()
        self.update_ref(self.tau_acs, self.tau_rw_real, self.tau_dist.abc)
        self.ads_step()

        self.i += 1
        if self.i % 4000 == 0:
            print(f"i={self.i}")

        return self.omega_true, self.q_true, self.acs.q_target, self.tau_mtq_real, self.m_mtq_real, self.tau_rw_real, self.tau_hybrid_real, self.sat.h_rw, self.B, self.sun, self.tau_dist, self.ads.esoq2.q_opt, self.q_st, self.ads.esoq2.cond_fro, self.ads.esoq2.cond_inf, self.ads.esoq2.cond_inf_neg, self.ads.esoq2.lambda_max, self.ads.esoq2.q_norm

    def ads_step(self):
        if self.i % self.dt_ratios == 0:

            omega_noisy = self.omega_true + NOISE_FLAG * 1e-3 * np.random.randn(3)

            if self.acs.cur_state is ACSState.NOMINAL:
                if self.i % (self.f_st * self.dt_ratios) == 0:
                    self.star_tracker_measure()
                else:
                    # if no measurement is available, bypass the estimate as a "pseudo-measurement"
                    # -> will not be taken into consideration
                    self.q_st = self.ads.q_pred


            if ESTIMATION in [EstimationConfig.UKF_KF_ST_ONLY, EstimationConfig.UKF_KF_ESQO]:
                self.ads.ukf_kf(omega_noisy, self.q_st, self.B, self.sun, self.tau_acs + INCLUDE_DIST_FLAG * self.tau_dist.abc_noisy,
                                self.sat, self.acs.cur_state, None)

                self.ads.ukf_full_state(omega_noisy, self.q_st, self.B, self.sun,
                                        self.tau_acs + INCLUDE_DIST_FLAG * self.tau_dist.abc_noisy, self.ukf_sat,
                                        self.omega_true)  # self.ads.omega_pred)

            elif ESTIMATION is EstimationConfig.UKF_DECOMPOSED:
                self.ads.ukf_decomposed(omega_noisy, self.q_st, self.B, self.sun,
                                        self.tau_acs + INCLUDE_DIST_FLAG * self.tau_dist.abc_noisy,
                                        self.sat, self.acs.cur_state)

            elif ESTIMATION is EstimationConfig.UKF_FULL_STATE:
                self.ads.ukf_full_state(omega_noisy, self.q_st, self.B, self.sun,
                                        self.tau_acs + INCLUDE_DIST_FLAG * self.tau_dist.abc_noisy, self.sat, None)

    def star_tracker_measure(self):
        # star tracker measurement
        # img = self.sat.star_tracker.synthesize(q_true.to_rotation())  # todo: kellene orientáció alapú keresés is
        # self.q_st = self.sat.star_tracker.predict(img)
        self.q_st = Quaternion(self.q_true.q + NOISE_FLAG * .0002 * np.random.randn(
            4))  # [0;.2] factor ~ 6.5° error/axis (Markley&Crassidis, 315.p); .000085 ~ 110" mean error in norm (3 sigma is appr. 260")

    def env_step(self):
        a, B_vec, _, _, _, _, _, sun_pos, _, dist_torques = self.env.step(self.sat.inertia.matrix)

        self.B = FrameVector.from_ref(B_vec, self.q_true, NOISE_FLAG * 3.5e-8 * np.random.randn(3))
        self.tau_dist = FrameVector.from_ref(dist_torques, self.q_true, NOISE_FLAG * 14e-7 * np.random.randn(3))
        self.sun = FrameVector.from_ref(sun_pos, self.q_true, NOISE_FLAG * 8e-3 * np.random.randn(3))

    def acs_step(self):
        if self.i % self.dt_ratios == 0:
            # set target attitude in case it is specified in ECEF
            self.acs.q_target = self.attitude_target.get_current_attitude(self.env.time)

            # calculate control torques
            self.tau_acs, self.tau_hybrid_real, self.tau_mtq_real, self.tau_rw, self.tau_rw_real, self.m_mtq, self.m_mtq_real = self.acs.update(
                self.B.abc, self.B.abc_noisy, self.sat.dynamics.x, self.ads.q_pred, self.sat.inertia.matrix,
                self.sat.h_rw)
            # self.tau_acs= self.tau_hybrid_real= self.tau_mtq_real= self.tau_rw= self.tau_rw_real= self.m_mtq= self.m_mtq_real = np.zeros(3)
            self.sat.h_rw += self.tau_rw_real * self.sat.dt

            if ESTIMATION in [EstimationConfig.UKF_KF_ST_ONLY, EstimationConfig.UKF_KF_ESQO]:
                self.ukf_sat.h_rw += self.tau_rw_real * self.ukf_sat.dt

    def update_ref(self, tau_acs, tau_rw_real, dist_torques_abc):
        self.sat_ideal.h_rw += tau_rw_real * self.sat_ideal.dt
        self.sat_ideal.dynamics.x = self.omega_true = self.sat_ideal.dynamics.predict(dist_torques_abc + tau_acs,
                                                                                      self.sat_ideal.inertia.matrix,
                                                                                      self.sat_ideal.dt,
                                                                                      self.sat_ideal.h_rw)

        # update kinematics with the true values
        self.sat_ideal.kinematics.A = SatelliteDescriptor.kinematics_with_bias(self.omega_true, np.zeros(3),
                                                                               self.sat_ideal.dt)

        self.sat_ideal.kinematics.x = self.sat_ideal.kinematics.predict(np.zeros(7))

        self.q_true = Quaternion(self.sat_ideal.kinematics.x[0:4], True)
        self.sat_ideal.kinematics.x[0:4] = self.q_true.q

        # update environmental vectors in the ABC frame with the new orientation
        self.sun.update_abc(self.q_true)
        self.B.update_abc(self.q_true)
        self.tau_dist.update_abc(self.q_true)


