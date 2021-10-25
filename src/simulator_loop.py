import numpy as np

from logger import Logger
from quaternion import Quaternion
from simulator import Simulator


class SimulatorLoop(object):

    @staticmethod
    def core(num_step: int, timestep: int = 250, t_sat_sample: int = 250, roll: float = .65, pitch: float = .37,
             yaw: float = 1.2, f_st: int = 4, omega_norm_deg=80, log_dir: str = "default", uncertain_inertia=False,
             nominal_only=False, detumbling_disturb=False, use_eci=None, use_ecef=None, fixed_init=True,
             tumble_off=False):
        """

        :param tumble_off:
        :param omega_norm_deg:
        :param fixed_init:
        :param use_ecef:
        :param use_eci:
        :param num_step: number of simulation steps
        :param timestep: timestep of numeric integration
        :param t_sat_sample: sample time of the ADCS in ms
        :param roll: roll angle in radians (x-axis)
        :param pitch: pitch angle in radians (y-axis)
        :param yaw: yaw angle in radians (z-axis)
        :param f_st: update frequency of the attitude q with the star tracker measurement
        :param log_dir: log directory
        :param uncertain_inertia: flag whether add noise to the inertia matrix to model uncertainty
        :param nominal_only: flag whether to use nominal control only
        :return:
        """

        # configure simulator and logger
        sim = Simulator(q_target=Quaternion.from_euler(yaw, pitch, roll), timestep=timestep, t_sat_sample=t_sat_sample,
                        f_st=f_st, omega_norm_deg=omega_norm_deg, uncertain_inertia=uncertain_inertia,
                        nominal_only=nominal_only, detumbling_disturb=detumbling_disturb, use_eci=use_eci,
                        use_ecef=use_ecef, fixed_init=fixed_init, tumble_off=tumble_off)
        logger = Logger(log_dir)

        # run the simulation
        log_frequency  = max(num_step // 20, 200)
        for i in range(num_step):
            omega, q_true, q_target, tau_mtq, m_mtq, tau_rw, tau_hybrid, h_rw_abc, B, sun, dist, q_esoq, q_st, esoq_cond_fro, esoq_cond_inf, esoq_cond_inf_neg, esoq_lambda_max, esoq_q_norm = sim.step()

            logger.log(
                # angular velocity
                omega=omega,
                omega_pred=sim.ads.omega_pred,

                # attitude (as Euler angles)
                angles=q_true.to_euler(),
                angles_pred=sim.ads.q_pred.to_euler(),
                angles_target=q_target.to_euler(),
                angles_ref_pred=sim.ads.ukf_full.q_pred.to_euler(),

                angles_esoq=q_esoq.to_euler(),
                esoq_cond_fro=esoq_cond_fro,
                esoq_cond_inf=esoq_cond_inf,
                esoq_cond_inf_neg=esoq_cond_inf_neg,
                esoq_lambda_max=esoq_lambda_max,
                esoq_q_norm=esoq_q_norm,
                angles_st=q_st.to_euler(),

                # errors (as Euler angles)
                ads_error_angles=sim.ads.q_pred.conj.quatprod(q_true).to_euler(),
                ads_ref_error_angles=sim.ads.ukf_full.q_pred.conj.quatprod(q_true).to_euler(),
                acs_error_angles=q_target.conj.quatprod(q_true).to_euler(),

                # hybrid torques
                hybrid_torques=tau_hybrid,

                # MTQs
                mtq_m=m_mtq,
                mtq_torques=tau_mtq,

                # RWs
                rw_torques=tau_rw,
                h_rw=h_rw_abc,

                # magnetic field
                magnet=B.ref,
                magnet_abc=B.abc,

                # Sun vector
                sun=sun.ref,
                sun_abc=sun.abc,

                # disturbance torques
                dist_torques=dist.ref,
                dist_torques_abc=dist.abc
            )

            if i % log_frequency == 0:
                logger.save()

        # save
        logger.save()

    @staticmethod
    def coarse(num_step: int = 150000, timestep: int = 250, t_sat_sample: int = 250, f_st: int = 4):
        step_deg = 5

        # iterate over the whole 3D polar coordinate space
        for yaw in range(-180, 181, step_deg):
            for pitch in range(-90, 91, step_deg):
                for roll in range(-180, 181, step_deg):
                    yaw = np.deg2rad(yaw)
                    pitch = np.deg2rad(pitch)
                    roll = np.deg2rad(roll)

                    # run the simulation loop with given initial parameters
                    SimulatorLoop.core(num_step, timestep, t_sat_sample, roll=roll, pitch=pitch, yaw=yaw, f_st=f_st,
                                       log_dir="course_sweep")

    @staticmethod
    def fine(num_step: int = 150000, timestep: int = 250, t_sat_sample: int = 250, f_st: int = 4):
        step_deg = 90

        # iterate over the critical parts of the 3D polar space
        for yaw in range(0, 361, step_deg):
            for pitch in range(-90, 91, step_deg):
                for roll in range(0, 361, step_deg):
                    if pitch == 0 and yaw == 0:
                        continue

                    fine_step = .05

                    # iterate over the environment of the critical parts with higher resolution
                    for yaw_fine in np.arange(0, 5.001, fine_step):
                        for pitch_fine in np.arange(0, 5.001, fine_step):
                            for roll_fine in np.arange(0, 5.001, fine_step):
                                yaw = np.deg2rad(yaw + yaw_fine)
                                pitch = np.deg2rad(pitch + pitch_fine)
                                roll = np.deg2rad(roll + roll_fine)

                                # run the simulation loop with given initial parameters
                                SimulatorLoop.core(num_step, timestep, t_sat_sample, roll=roll, pitch=pitch, yaw=yaw,
                                                   f_st=f_st, log_dir="fine_sweep")

    @staticmethod
    def monte_carlo(num_runs: int = 10, num_step: int = 150000, timestep: int = 250, t_sat_sample: int = 250,
                    f_st: int = 4, omega_norm_deg=80, log_dir: str = "mc", uncertain_inertia: bool = False,
                    detumbling_disturb: bool = False, nominal_only: bool = False, use_eci=None, use_ecef=None,
                    fixed_init=True, tumble_off=False):
        for i in range(num_runs):
            print("------------------------------")
            print(f"Run number {i+1}/{num_runs}")
            print("------------------------------\n")
            # run the simulation loop with given initial parameters
            SimulatorLoop.core(num_step, timestep, t_sat_sample, f_st=f_st, omega_norm_deg=omega_norm_deg,
                               log_dir=log_dir, uncertain_inertia=uncertain_inertia, nominal_only=nominal_only,
                               detumbling_disturb=detumbling_disturb, use_eci=use_eci, use_ecef=use_ecef,
                               fixed_init=fixed_init, tumble_off=tumble_off)