from enum import Enum

import numpy as np

from alg_utils import cross_prod_mat
from quaternion import Quaternion


class Controller(object):
    def __init__(self) -> None:
        super().__init__()

    def update(self, **kwargs):
        raise NotImplementedError(
            f"The {self.update.__name__}  function should be implemented after subclassing {self.__class__}!")


class DetumblingControl(Controller):

    def __init__(self, m_func, ctrl_gain=5e-3) -> None:
        super().__init__()
        self.m_func = m_func
        self.ctrl_gain = ctrl_gain

        self.B_last = None  # np.array([5e-5, 0, 0])

    def update(self, B_model, B_meas, omega):
        """

        :param B_model: magnetic field vector in ABC frame given by the IGRF model
        :param B_meas: measured magnetic field vector in ABC frame
        :param omega: angular velocity in the ABC frame
        :return:
        """

        # if self.B_last is None:
        #     self.B_last = B_meas

        B_pred_norm_inv = 1. / np.linalg.norm(B_meas)
        b_pred = B_meas * B_pred_norm_inv

        # dipole moment
        # self.ctrl_gain = 5e-3
        m = self.ctrl_gain * B_pred_norm_inv * np.cross(omega, b_pred)
        m_real = self.m_func(m)

        # self.ctrl_gain = -3500000
        # dt = .6
        # B_dot = (B_meas - self.B_last)/dt
        # self.B_last = B_meas
        # m2 = -self.ctrl_gain*B_dot

        # torque
        # todo: m should be commanded to the MTQ, which returns with a torque (incl. saturating u)
        tau_det = np.cross(m, B_model)  # here the true magnetic field should be used
        # tau_det2 = np.cross(m2, B_model)  # here the true magnetic field should be used

        # print(f"tau_det={tau_det}, b_dot={B_dot}")

        # thr = 5e-4
        # if np.linalg.norm(tau_det2) > thr:
        #     tau_det2 = tau_det2 / np.linalg.norm(tau_det2) *thr
        # breakpoint()

        # print(f"tau_det={tau_det}, tau_det2={tau_det2}")
        return tau_det, m, m_real


class StaticInputAllocationControl(Controller):
    def __init__(self, q_target: Quaternion, omega_target: np.ndarray, dt, h_ref: np.ndarray, m_func: callable,
                 rw_func: callable, hybrid_c: float = 3e-3,
                 hybrid_delta: float = .7, dump_gain: float = 5e-3) -> None:
        """

        :param dt: sample time
        :param q_target: target attitude quaternion in ECI frame
        :param omega_target: target angular velocity in ABC frame
        :param h_ref: reference RW angular momentum in ABC frame
        :param hybrid_c: c coefficient of the hybrid controller
        :param hybrid_delta: delta coefficient of the hybrid controller
        :param dump_gain: gain of the momentum dumping controller
        :param m_func:
        :param rw_func:
        """
        super().__init__()
        self.dt = dt

        """Hybrid attitude control"""
        self.q_target = q_target
        self.omega_target = omega_target
        self.c = hybrid_c
        self.delta = hybrid_delta
        self.x_c = 1
        self.m_func = m_func
        self.rw_func = rw_func

        """Momentum dumping"""
        self.h_ref = h_ref
        self.dump_gain = dump_gain
        # self.dump_gain_d = 0*8e-0
        # self.dump_gain_i = 3e-5
        # self.h_err_last = np.zeros(3)
        # self.h_err_integral = np.zeros(3)

    def update(self, B_abc: np.ndarray, omega_abc: np.ndarray, q_est: Quaternion, inertia: np.ndarray, m_det, h_rw):
        """

        :param B_abc: measured magnetic field vector in ABC frame
        :param omega_abc: angular velocity in the ABC frame
        :param q_est: estimated attitude quaternion in ECI frame
        :param inertia: inertia matrix of the satellite
        :return:
        """
        # hybrid reference
        tau_hybrid = self._hybrid_ac(q_est, inertia, omega_abc)

        # MTQs
        m_mtq = self._momentum_dumping(B_abc, h_rw)  # reference



        # todo: mi van akkor ha nem csak lecsapjuk a szaturált értékeket, hanem az egész vektort átskálázzuk
        #  , hogy legalább az iránya jó legyen,?

        m_mtq_real = self.m_func(m_mtq)  # real

        tau_mtq_real = np.cross(m_mtq_real, B_abc)  # torque


        # todo: better naming shall be chosen
        # m_mtq_real  = self.m_func(self._mtq_control(tau_hybrid, tau_mtq_real, B_abc, m_mtq_real))
        # tau_mtq_real = np.cross(m_mtq_real, B_abc)

        # RW
        tau_rw = self._rw_control(tau_hybrid, B_abc, tau_mtq_real, omega_abc, h_rw)  # refernce

        tau_rw_real = self.rw_func(tau_rw)  # real

        # print(f"m_diff={np.linalg.norm(m_mtq - m_mtq_real)}, rw_diff={np.linalg.norm(tau_rw-tau_rw_real)}")

        # hybrid real
        tau_hybrid_real = self._real_hybrid_torque(tau_mtq_real, tau_rw_real, B_abc, omega_abc, h_rw)


        # if np.linalg.norm(tau_hybrid - tau_hybrid_real) < 1e-12:
        #     tau_hybrid_real = tau_hybrid

        # print(f"tau_h_diff={np.linalg.norm(tau_hybrid - tau_hybrid_real)}")
        return tau_hybrid_real, tau_mtq_real, tau_rw, tau_rw_real, m_mtq, m_mtq_real

    def _momentum_dumping(self, B: np.ndarray, h_rw:np.ndarray) -> np.ndarray:
        """

        :param B: :param B: magnetic field vector in ABC frame
        :return:
        """
        h_err = h_rw - self.h_ref
        # h_err_d = (h_err - self.h_err_last)/self.dt
        # antiwindup_limit = .5
        # cond = (self.h_err_integral > antiwindup_limit)*(np.sign(h_err) == -1) + (self.h_err_integral < -antiwindup_limit)*(np.sign(h_err) == 1) +(np.abs(self.h_err_integral) < antiwindup_limit)
        # self.h_err_integral += cond*h_err*self.dt
        # print(self.h_err_integral)
        m_dump_mtq = -1. / np.linalg.norm(B) ** 2 *  cross_prod_mat(B) @ (self.dump_gain *h_err)# + self.dump_gain_i*self.h_err_integral + self.dump_gain_d*h_err_d)
        # tau_dump_mtq = np.cross(m_dump_mtq, B)# here the R(q) before [Bx] is included in B_abc (the norm is the same in every coordinate system)

        # self.h_err_last = h_err
        return m_dump_mtq

    # def _mtq_control(self, tau_hybrid: np.ndarray, tau_mtq: np.ndarray,B: np.ndarray, m_mtq:np.ndarray):
    #     tau_residual = tau_hybrid-tau_mtq
    #     m_direction = np.cross(B, tau_residual)
    #     m_residual =  m_direction *.3*np.linalg.norm(m_mtq)/np.linalg.norm(m_direction) # scale m_residual to be the same length as the momentum dumping m
    #
    #     return m_residual+m_mtq


    def _rw_control(self, tau_hybrid: np.ndarray, B: np.ndarray, tau_mtq: np.ndarray, omega: np.ndarray, h_rw:np.ndarray) -> np.ndarray:
        """

        :param tau_hybrid: hybrid controller torque in ABC frame
        :param B: magnetic field vector in ABC frame
        :param tau_mtq: MTQ torque in ABC frame
        :param omega: angular velocity in the ABC frame
        :return:
        """
        tau_rw = -tau_hybrid + tau_mtq - cross_prod_mat(omega) @ h_rw
        return tau_rw

    def _hybrid_ac(self, q_est: Quaternion, inertia: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Source: Robust Global Asymptotic Attitude Stabilization of a Rigid Body by Quaternion-based Hybrid Feedback


        :param q_est: estimated attitude quaternion in ECI frame
        :param inertia: inertia matrix of the satellite
        :param omega: angular velocity in the ABC frame
        :return:
        """
        # calculate the error quaternion
        # q_err = q_est.quatprod(self.q_target.conj)
        q_err = self.q_target.conj.quatprod(q_est)
        # q_err = Quaternion.from_sw(q_est.scalar-self.q_target.scalar, q_est.vector-self.q_target.vector, False)
        # print(q_err.q)

        # discrete
        if q_err.scalar * self.x_c < -self.delta:
            self.x_c *= -1
            print("---------------------switched---------------------")

        # continuous

        # omega_norm = np.linalg.norm(omega)
        # omega_norm = min(omega_norm, np.deg2rad(4))
        #
        # factor = 0.5
        # weight = factor* (omega_norm / np.deg2rad(4) if omega_norm > np.deg2rad(2) else 0.)
        # sum_weight = np.exp(weight) + np.exp(factor - weight)

        # tau = -self.c * self.x_c * q_err.vector * np.exp(1-weight) / sum_weight - inertia @ (omega - self.omega_target) *np.exp(weight) / sum_weight
        tau = -self.c * self.x_c * q_err.vector - 85e-3 * (omega - self.omega_target)

        return tau

    def _real_hybrid_torque(self, tau_mtq: np.ndarray, tau_rw: np.ndarray, B: np.ndarray, omega: np.ndarray, h_rw:np.ndarray):
        """
        Calculate the real hybrid torque based on the actuator measurements

        :param tau_mtq: MTQ torque in ABC frame
        :param tau_rw: RW torque in ABC frame
        :param B: magnetic field vector in ABC frame
        :param omega: angular velocity in the ABC frame
        :return:
        """
        tau_hybrid_real = -tau_rw + tau_mtq - cross_prod_mat(omega) @ h_rw

        return tau_hybrid_real


class ActuatorSentinel(object):

    def __init__(self, alpha: float = .4, error_threshold: float = 1e-1, value_threshold: float = 5) -> None:
        super().__init__()
        self.alpha = alpha
        self.error_threshold = error_threshold
        self.value_threshold = value_threshold

        self.value_exp_avg = np.zeros(3)
        self.abs_err_exp_avg = np.zeros(3)

    def update(self, value: np.ndarray, value_ref: np.ndarray) -> None:
        self.abs_err_exp_avg = self.alpha * self.abs_err_exp_avg + (1 - self.alpha) * np.abs(value - value_ref)
        self.value_exp_avg = self.alpha * self.value_exp_avg + (1 - self.alpha) * value

    def _calc_ref_component(self, ref: np.ndarray, idx: int) -> float:

        if self.abs_err_exp_avg[idx] > self.error_threshold:

            if self.value_exp_avg[idx] < self.value_threshold:
                ref = 0
            else:
                ref[idx] = self.value_exp_avg[idx]

        return ref

    def calc_ref(self, ref: np.ndarray) -> np.ndarray:
        ref_x = self._calc_ref_component(ref, 0)
        ref_y = self._calc_ref_component(ref, 1)
        ref_z = self._calc_ref_component(ref, 2)

        return np.array([ref_x, ref_y, ref_z])


class ACSState(Enum):
    NO_CONTROL = 0
    DETUMBLING = 1
    NOMINAL = 2


class ACS(object):
    def __init__(self, q_target: Quaternion, omega_target: np.ndarray, dt: float, h_ref: np.ndarray, m_func: callable,
                 rw_func: callable, det_threshold_enter: float = np.deg2rad(3), det_threshold_exit=np.deg2rad(1.2),
                 hybrid_c: float = 3e-3, hybrid_delta: float = .5, dump_gain: float = 5e-3,
                 detumbling_gain: float = 8e-3, nominal_only=False) -> None:
        super().__init__()

        # flag whether use only SIA
        self.nominal_only = nominal_only

        # thresholds
        self.det_threshold_enter = det_threshold_enter
        self.det_threshold_exit = det_threshold_exit

        """Controllers"""
        self.detumbling = DetumblingControl(m_func, detumbling_gain)
        self.sia = StaticInputAllocationControl(q_target, omega_target, dt, h_ref, m_func, rw_func, hybrid_c,
                                                hybrid_delta, dump_gain)

        """Actuator supervision"""
        self.mtq_sentinel = ActuatorSentinel()
        self.rw_sentinel = ActuatorSentinel()

        """State machine"""
        self.cur_state = ACSState.NOMINAL
        self.req_state = ACSState.NOMINAL
        self.det_cntr = 0

    @property
    def q_target(self):
        return self.sia.q_target

    @q_target.setter
    def q_target(self, val):
        self.sia.q_target = val

    def update(self, B_model: np.ndarray, B_abc: np.ndarray, omega_abc: np.ndarray, q_est: Quaternion,
               inertia: np.ndarray, h_rw:np.ndarray):
        pass

        """Autonomous checks"""
        omega_norm = np.linalg.norm(omega_abc)
        if omega_norm > self.det_threshold_enter:
            if self.cur_state != ACSState.DETUMBLING:
                print(ACSState.DETUMBLING)

            self.cur_state = ACSState.DETUMBLING

        elif self.cur_state == ACSState.DETUMBLING and omega_norm < self.det_threshold_exit:  # and self.det_cntr >500:
            if self.cur_state != ACSState.NOMINAL:
                print(ACSState.NOMINAL)

            self.cur_state = self.req_state

        # else:
        #     if self.cur_state == ACSState.DETUMBLING and self.req_state != self.cur_state:
        #         self.cur_state = self.req_state

        """Handle control modes"""
        tau = tau_hybrid_real = tau_mtq_real = tau_rw = tau_rw_real = m_mtq = m_mtq_real = m_det = np.zeros(3)
        if not self.nominal_only and self.cur_state == ACSState.DETUMBLING or self.req_state == ACSState.DETUMBLING:
            """DETUMBLING"""
            # self.det_cntr +=1
            # print(f"detumbling, {self.det_cntr}")
            # print(f"detumbling")
            self.cur_state = ACSState.DETUMBLING

            tau, m_mtq, m_mtq_real = self.detumbling.update(B_model, B_abc, omega_abc)
            tau_mtq_real = tau
            m_det = m_mtq
            self.mtq_sentinel.update(m_mtq_real, m_mtq)  # m <- I

        elif self.cur_state == ACSState.NOMINAL or self.req_state == ACSState.NOMINAL:
            """NOMINAL"""
            # print("*********************nominal*************")
            self.cur_state = ACSState.NOMINAL
            tau_hybrid_real, tau_mtq_real, tau_rw, tau_rw_real, m_mtq, m_mtq_real = self.sia.update(B_abc, omega_abc,
                                                                                                    q_est, inertia,
                                                                                                    m_det,h_rw)

            # tau = tau_hybrid_real
            tau = tau_mtq_real - tau_rw_real # h_rw will be passed to the dynamic model directly

            self.mtq_sentinel.update(m_mtq_real, m_mtq)  # m <- I
            self.rw_sentinel.update(tau_rw_real, tau_rw)  # tau <- omega <- u

        # else:
        #     """NO_CONTROL"""
        #     self.cur_state = ACSState.NO_CONTROL
        #     tau = np.zeros(3)

        # print(self.mtq_sentinel.abs_err_exp_avg, self.rw_sentinel.abs_err_exp_avg)

        return tau, tau_hybrid_real, tau_mtq_real, tau_rw, tau_rw_real, m_mtq, m_mtq_real

    @property
    def state(self):
        return self.cur_state

    @state.setter
    def state(self, value):
        self.req_state = value
