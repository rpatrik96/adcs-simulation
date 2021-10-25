from builtins import object

import numpy
import numpy as np

from alg_utils import separated_chol_update
from model import Model, LinearModel
from quaternion import Quaternion


def scale_magnet_sun(magnet_sun_inner_prod, offset=.9, scale=40):
    inner_prod_scaled = (np.exp(scale * (abs(magnet_sun_inner_prod) + offset)) - np.exp(scale * offset)) / (
            np.exp(scale * (1 + offset)) - np.exp(scale * offset))
    return inner_prod_scaled

def inv_clip_esoq_q_norm(esoq_q_norm, q_norm_thresh=1.0):
    return 1 - np.clip(esoq_q_norm, 0, q_norm_thresh)

class KalmanFilter(object):
    def __init__(self, model: LinearModel, Q: np.ndarray, R: np.ndarray, factor=90) -> None:
        """
        Kalman-filter class


        Notation: - hat: prediction
                  - bar: propagation

        :param model: a LinearModel instance describing the system dynamics
        :param Q: process/system noise covariance matrix
        :param R: measurement noise covariance matrix
        """
        super().__init__()
        self.factor = factor
        self.model = model
        self.dim = self.model.A.shape[0]
        self.Q = Q  # system/process noise covariance
        self.R = R  # measurement noise covariance
        self.alpha = .9
        self.cov_weight = 0.  # ESOQ2 vector parallelism
        self.cov_norm_penalty = 0.

        self.P_pred = 1e6 * np.eye(self.dim)
        self.x_pred = np.zeros(self.dim)

    def update(self, u, y):
        """
        Complete update of the Kalman-filter

        Notation: - K: Kalman gain
                  - S: innovation covariance matrix

        :param u: input vector
        :param y: measurement
        :return:
        """
        """Time update"""  # system dynamics
        # predict the state with the system model
        x_hat = self.model.predict(u)
        # propagate covariance - it will be increased by two components:
        # 1. the transformed covariance matrix.
        # Note that APA' is nothing else than A(x-E[x])(x-E[x])'A'= {A(x-E[x])} {A(x-E[x])}',
        # which is the covariance of the transformed variable Ax
        # 2. the system/process noise (model inaccuracies not accounted for, e.g. not modeled effects, linearizations)
        P_hat = self.model.A @ self.P_pred @ self.model.A.T + self.Q

        """Measurement update"""  # incorporate measurement
        # temporary variables for accelerating computation
        R_corrected = self.R_corrected()
        P_hat_C_tr = P_hat @ self.model.C.T
        S_without_R = self.model.C @ P_hat_C_tr

        # Propagation of the covariance uncertainty toward the observation/output quantity - increased by two components as well
        # Note that implicitly here we have three components
        # 1. the (again) transformed covariance matrix
        # the first variable in S equals C P_hat C' = C (APA' + Q) C' = CAP(CA)' + CQC' + R, where
        # CAP(CA)' is the covariance of variable CAx
        # CQC' is the covariance accounting to system inaccuracies, mapped to the observations
        # R the covariance of the observation noise
        # 2. the observation noise (e.g. due to sensor noise)
        S = S_without_R + R_corrected

        # The Kalman gain is the optimal coefficient for incorporating measuement to correct the prediction (done in the time update)
        # It can be derived as minimizing the MSE of the state and its prediction
        # K can be written as
        #           (APA' + Q)C'
        # K = ----------------------
        #       C(APA' + Q)C' + R
        # i.e., if the observation is noisy (R is big), then K will be small, but if the model is not quite good (Q is big)
        # and/or the covariance of the state is big (i.e., P), then K will be appr. 1
        # Note that the Kalman gain will be always in (0;1] - in the scalar case. Thus, it has a contractive effect.
        K = P_hat_C_tr @ np.linalg.inv(S)

        # The Kalman gain is used for correcting the covariance as well
        # (the covariance P_hat predicted in the time update can be "reduced" by the new information of the observations)
        # A more intuitive (but computationally more burdensome) variant for the posterior covariance is
        # P = (I-KC) P_hat (I-KC)' + KRK'
        # This can be intuitively thought of as somehow trying to decrease the propoagated covariance P_hat and
        # the effect of the measurement noise covariance R
        self.P_pred = (np.eye(self.dim) - K @ self.model.C) @ P_hat

        # Innovation is an important concept in the theory of Kalman filtering.
        # This is the residual (basically, the new information), the model was not able to explain.
        # Thus, this should be incorporated into the posterior state
        innovation = y - self.model.C @ x_hat

        # To do that, we need to transform somehow from the observation space into the state space.
        # This is done with the Kalman gain.
        weighted_innovation = K @ innovation

        # Posterior state update
        # This can be written as : x_hat + K(y-Cx_hat) = (I-KC)x_hat + Ky, and this expression shed light on the form of
        # the posterior covariance, as that is constructed by the covariance of (I-KC)x_hat - this is  (I-KC) P_hat (I-KC)' -,
        # and by the covariance of Ky, which is KRK'
        self.x_pred = x_hat + weighted_innovation

        # residual = y - self.model.C @ self.x_pred

        # self._cov_update(residual, weighted_innovation, S_without_R)

    def _cov_update(self, residual, weighted_innovation, S_without_R):
        # covariance update as in
        # Adaptive Adjustment of Noise Covariance in Kalman Filter for Dynamic State Estimation
        # https://arxiv.org/pdf/1702.00884
        self.R = self.alpha * self.R + (1. - self.alpha) * (np.outer(residual, residual) + S_without_R)
        self.Q = self.alpha * self.Q + (1. - self.alpha) * np.outer(weighted_innovation, weighted_innovation)

    def R_corrected(self):
        covariance = self.R[0, 0]
        val = covariance * self.factor

        norm_penalty = inv_clip_esoq_q_norm(self.cov_norm_penalty)
        inner_prod_scaled = scale_magnet_sun(self.cov_weight)

        return self.R + inner_prod_scaled * norm_penalty * np.diagflat([0, 0, 0, 0, val, val, val, val])


class UnscentedTransform(object):

    def __init__(self, f: callable, is_error, dim, w0: float = 0.05, S_scale=10.9, linear: bool = False,
                 redraw_sigma: bool = True) -> None:
        super().__init__()
        self.f = f
        self.redraw_sigma = redraw_sigma
        self.linear = linear
        self.S_scale = S_scale
        self.w0 = w0
        self.is_error = is_error
        self.dim = dim

        if self.is_error:
            self.q_pred = Quaternion()
            self.omega_pred = np.zeros(3, )

        self._symm_sigma_coeffs()

    def __call__(self, mean, S_cov, noise_cov_sqrt, **kwargs):
        sigma_points = self._symm_sigma_set(S_cov, mean) if self.redraw_sigma is True else mean
        if not self.linear:
            sigma_points = self._propagate_sigma_points(self.f, sigma_points, **kwargs)
        mean_hat = self._calc_mean_sigma_points(sigma_points)  # .reshape(-1, 1)

        # sigma point conversion for the error state formulation
        (mean_hat, sigma_points, q_pred) = (
            mean_hat, sigma_points, None) if self.is_error is False else self._calc_error_sigma_points(
            mean_hat, sigma_points)

        S_hat, S_pos, S_neg = self._update_sqrt_cov(mean_hat, noise_cov_sqrt, sigma_points)

        return S_hat, S_pos, S_neg, sigma_points, mean_hat, q_pred

    def _propagate_sigma_points(self, f, sigma_points, **kwargs):
        return np.array([f(sigma_points[:, i], **kwargs) for i in range(self.num_sigma_points)]).T

    def _convert_error_sigma_points2full_states(self, error_sigma_points):
        full_sigma_points = np.array([np.hstack([self._propagate_quaternion_error(sp[0:3]).q, sp[3:]])
                                      for sp in error_sigma_points.T]).T
        return full_sigma_points

    def _convert_full_sigma_points2error_states(self, q_pred, sigma_propagated):
        error_sigma = np.array([np.hstack([Quaternion(sp[0:4]).quatprod(q_pred.conj).vector,
                                           sp[4:]]) for sp in sigma_propagated.T]).T
        return error_sigma

    def _calc_error_sigma_points(self, mean_hat, sigma_propagated):
        mean_hat[0:4] /= np.linalg.norm(mean_hat[0:4])  # needed for the return value
        q_pred = Quaternion(mean_hat[0:4], True)
        error_sigma = self._convert_full_sigma_points2error_states(q_pred, sigma_propagated)
        error_mean_hat = self._calc_mean_sigma_points(error_sigma)
        return error_mean_hat, error_sigma, q_pred

    def _propagate_quaternion_error(self, q_err: np.ndarray, q_pred=None) -> Quaternion:
        return Quaternion.from_error_vec(q_err).quatprod(self.q_pred if q_pred is None else q_pred)

    def _update_sqrt_cov(self, mean_hat, noise_cov_sqrt, sigma_propagated):
        pos_sigma_points, neg_sigma_points = self._separate_sigma_points(sigma_propagated, mean_hat)

        S_hat = separated_chol_update(pos_sigma_points, neg_sigma_points, noise_cov_sqrt)
        return S_hat, pos_sigma_points, neg_sigma_points

    def _calc_mean_sigma_points(self, sigma_propagated):
        return sigma_propagated @ self.weights

    def _symm_sigma_coeffs(self):
        self.num_sigma_points = 2 * self.dim + 1

        # homogeneous, minimum symmetrics sigma set coefficients (HMT thesis, p. 90 in the pdf)
        self.weights = np.array([(1 - self.w0) / (2 * self.dim)] * self.num_sigma_points)
        self.weights[0] = self.w0

    def _symm_sigma_set(self, S_cov: np.ndarray, mean: np.ndarray):
        if self.S_scale is None:
            self.S_scale = 10.9 if not self.is_error else 0.25  # 1./np.sqrt(self.weights[-1])

        mean = mean.reshape(-1, 1)
        mean = mean if self.is_error is False else np.zeros_like(mean)  # mean is zero for error formulation
        sigma_1_n = mean + self.S_scale * S_cov
        sigma_n_2n = mean - self.S_scale * S_cov

        sigma_points = np.concatenate((mean, sigma_1_n, sigma_n_2n), axis=-1)
        sigma_points = sigma_points if self.is_error is False else self._convert_error_sigma_points2full_states(
            sigma_points)

        return sigma_points

    def _separate_sigma_points(self, sigma_points: np.ndarray, mean: np.ndarray):
        """

        :param sigma_points:
        :param mean:
        :return: pos_sigma_points, neg_sigma_points (S_pos, S_neg in the HMT thesis)
        """
        pos_idx = self.weights >= 0
        neg_idx = self.weights < 0

        mean = mean.reshape(-1, 1)
        pos_sigma_points = np.sqrt(self.weights[pos_idx]) * (sigma_points[:, pos_idx] - mean)
        neg_sigma_points = np.sqrt(np.abs(self.weights[neg_idx])) * (sigma_points[:, neg_idx] - mean)

        return pos_sigma_points, neg_sigma_points


class SquareRootSigmaPointKalmanFilter(object):

    def __init__(self, model: Model, Q: np.ndarray, R: np.ndarray, is_error=False) -> None:
        """

        Notation: - hat: prediction
                  - bar: propagation

        :param is_error:
        :param Q:
        :param R:
        """
        super().__init__()
        self.model = model
        self.dim = Q.shape[0]
        self.is_error = is_error

        w0 = 0.15
        s_scale = 1.9
        self.f_ut = UnscentedTransform(self.model.f, is_error, self.dim, w0,
                                       s_scale if self.is_error is False else np.array([0.25, 0.25, 0.25, 2, 2, 2]),
                                       linear=False, redraw_sigma=True)
        self.h_ut = UnscentedTransform(self.model.h, is_error, self.dim, w0,
                                       s_scale if self.is_error is False else np.array([0.25, 0.25, 0.25, 2, 2, 2]),
                                       linear=True if self.is_error is False else False,
                                       redraw_sigma=False if self.is_error is False else True)

        self.timestep = 0
        self.forget_factor = .7

        self.S_Q = np.linalg.cholesky(Q)
        self.R = R
        self.S_R = np.linalg.cholesky(R)

        self.S_pred = (3e1 if not self.is_error else np.diagflat([[3e-1] * 3, [3e1] * 3])) * np.eye(self.dim)
        self.x_pred = np.zeros(self.dim)
        # self.num_mc_sigma_points = 20

    def _calc_posterior_estimate(self, K, x_hat, y, y_hat, q_obs_pred):
        # todo: quaternion measurement should be converted to a vector for the ST

        if self.is_error:
            y_unrolled = np.hstack([v for v in y.values()])
            y_quat = Quaternion(y_unrolled[0:4], True)
            y_q = y_quat.quatprod(q_obs_pred.conj)

            innovation = np.hstack((y_q.vector, y_unrolled[4:] - y_hat[3:]))

            # # y_hat contains the quaternion obs as a vector; thus we need everything from index 3 on
            # # (only first 3 elements should be left out)
            # y_hat = np.hstack((q_obs_pred.q, y_hat[3:]))
        else:
            innovation = y - y_hat

        self.x_pred = x_hat + K @ innovation
        # print(f"omega_y={y}; omega_pred={self.x_pred}")
        # print(K@innovation)
        # print(f"omega_y={y['dynamics']}; omega_pred={self.x_pred[3:]}")

    def _calc_posterior_sqrt_cov(self, K, S_pos_hat, S_neg_hat, S_y_pos_hat, S_y_neg_hat):
        self.S_pred = separated_chol_update(S_pos_hat - K @ S_y_pos_hat, S_neg_hat - K @ S_y_neg_hat,
                                            K @ self.S_R)

    def _construct_cross_cov(self, S_pos, S_neg, S_y_pos, S_y_neg):
        """
        HMT thesis, p.105 (it is not used in LibMenegaz)
        As it seems, this kind of update is not even used in the code. Probably, that was only
        a way to introduce the formula for the square root of the state covariance.

        :param S_pos:
        :param S_neg:
        :param S_y_pos:
        :param S_y_neg:
        :return:
        """
        return S_pos @ S_y_pos.T - S_neg @ S_y_neg.T

    def _calc_kalman_gain(self, S_pos, S_neg, S_y_pos, S_y_neg, S_y_hat):
        P_xy = self._construct_cross_cov(S_pos, S_neg, S_y_pos, S_y_neg)
        return P_xy @ np.linalg.inv(S_y_hat @ S_y_hat.T)

    def update(self, u, y, f_kwargs=None, h_kwargs=None):
        # unscented transform

        if h_kwargs is None:
            h_kwargs = dict()
        if f_kwargs is None:
            f_kwargs = dict()

        S_hat, S_pos_hat, S_neg_hat, sigma_propagated, x_hat, q_state_pred = self.f_ut(self.x_pred, self.S_pred,
                                                                                       self.S_Q, u=u, **f_kwargs)

        if self.is_error:
            self.q_pred = q_state_pred  # to fulfil the // equation in HMT thesis, 166.p.

        S_y_hat, S_y_pos_hat, S_y_neg_hat, sigma_y_propagated, y_hat, q_obs_pred = self.h_ut(
            sigma_propagated if not self.is_error else x_hat, None if not self.is_error else S_hat, self.S_R,
            **h_kwargs)

        # Kalman gain
        K = self._calc_kalman_gain(S_pos_hat, S_neg_hat, S_y_pos_hat, S_y_neg_hat, S_y_hat)

        # posterior update
        self._calc_posterior_estimate(K, x_hat, y, y_hat, q_obs_pred)
        self._calc_posterior_sqrt_cov(K, S_pos_hat, S_neg_hat, S_y_pos_hat, S_y_neg_hat)

        self._update_pred_values(q_state_pred)
        # self.omega_pred = y["dynamics"]
        # self.q_pred = Quaternion(y["kinematics"][:4])

    def _update_pred_values(self, q_mean_hat):
        q_last_idx = 4 if not self.is_error else 3
        q_raw = self.x_pred[0:q_last_idx]
        self.q_pred = Quaternion(q_raw, True) if self.is_error is False else self.f_ut._propagate_quaternion_error(
            q_raw, q_mean_hat)

        # print(f"q_pred={q_raw}, omega_pred={omega}")
        # print(f"q_pred={self.q_pred}, omega_pred={self.omega_pred}")
        # print(self.omega_pred - self.x_pred[3:])
        self.omega_pred = self.x_pred[q_last_idx:]

        if self.is_error:
            self.x_pred[:3] = self.q_pred.vector

    @property
    def q_pred(self):
        return self.f_ut.q_pred

    @property
    def omega_pred(self):
        return self.f_ut.omega_pred

    @q_pred.setter
    def q_pred(self, value):
        self.f_ut.q_pred = self.h_ut.q_pred = value

    @omega_pred.setter
    def omega_pred(self, value):
        self.f_ut.omega_pred = self.h_ut.omega_pred = value

    # def _sample_random_sigma_set(self, S_cov: np.ndarray, mean: np.ndarray):
    #     """
    #
    #     :param S_cov: Cholesky-factor of the covariance matrix
    #     :param mean: mean of the state vector
    #     :return:
    #     """
    #     # alpha = 1.1
    #     # kappa = 0.
    #     # lambda_coef = alpha ** 2 * (self.dim + kappa) - self.dim
    #     # self.w_coeffs = np.array([.5 * lambda_coef / (lambda_coef + self.dim)] * (self.dim + 1))
    #     # self.w_coeffs[0] *= 2.
    #     self.mc_coeffs = self.weights = np.ones(self.num_mc_sigma_points) / self.num_mc_sigma_points
    #     return np.random.multivariate_normal(mean, S_cov @ S_cov.T, self.num_mc_sigma_points).T
    #
    # def _minimal_sigma_set(self, S_cov: np.ndarray, mean: np.ndarray):
    #     """
    #
    #     :param S_cov: Cholesky-factor of the covariance matrix
    #     :param mean: mean of the state vector
    #     :return:
    #     """
    #     kappa = 0.6
    #     factor = 1  # np.sqrt(self.dim + kappa)
    #     sigma_0 = - factor * S_cov @ (self.alpha * np.ones(self.dim)) / np.sqrt(self.weights[0]) + mean
    #     sigma_1_n = factor * S_cov @ self.C @ self.W_sqrt_inv + mean
    #     return np.concatenate((sigma_0.reshape(-1, 1), sigma_1_n), axis=-1)
    #
    # def _minimal_sigma_set_general(self, S_cov: np.ndarray, mean: np.ndarray):
    #     """
    #
    #     :param S_cov: Cholesky-factor of the covariance matrix
    #     :param mean: mean of the state vector
    #     :return:
    #     """
    #     E = 1 / np.sqrt(self.weights[0]) * S_cov @ self.C @ self.W_sqrt_inv
    #     sigma_0 = - E / self.weights[0] @ self.weights[1:] + mean
    #     sigma_1_n = E + mean
    #     return np.concatenate((sigma_0.reshape(-1, 1), sigma_1_n), axis=-1)
    #
    # def _minimal_sigma_coeffs(self):
    #     # self.kappa = .5
    #     w0 = .3  # self.kappa / (self.dim + self.kappa)
    #     self.num_sigma_points = self.dim + 1
    #     alpha_squared = (1 - w0) / self.dim
    #     ones = np.ones((self.dim, self.dim))
    #
    #     self.alpha = np.sqrt(alpha_squared)
    #     self.C = np.linalg.cholesky(np.eye(self.dim) - alpha_squared * ones)
    #     C_inv = np.linalg.inv(self.C)
    #     # col_sums = np.sum(C_inv, 1)  # the same as np.sum(C_inv.T, 0)
    #     #
    #     # w1_row = self.w_coeffs[0] * self.alpha ** 2 * col_sums
    #     # w_sqrt_coeffs = [np.sqrt(w1_row[0])]
    #     # we can start from 1, as np.sqrt(w1) is already in w_sqrt_coeffs
    #     # for i in range(1, self.dim):
    #     #     w_sqrt_coeffs.append(w1_row[i] / w_sqrt_coeffs[0])
    #     W_coeff_mat = w0 * alpha_squared * C_inv @ ones @ C_inv.T
    #     self.weights = np.array([w0] + [W_coeff_mat[i, i] for i in range(self.dim)])
    #     print(self.weights)
    #     self.W_sqrt_inv = np.linalg.inv(np.diag([np.sqrt(wi) for wi in self.weights[1:]]))
    #
    # def _minimal_sigma_coeffs_general(self):
    #     self.num_sigma_points = self.dim + 1
    #     self.v = np.ones(self.dim) / self.dim
    #     w0 = 1. / (1. + self.v @ self.v)
    #     self.weights = np.concatenate(([w0], w0 * self.v * self.v), axis=-1)
    #
    #     self.C = np.linalg.inv(np.linalg.cholesky(np.eye(self.dim) + np.outer(self.v, self.v)))
    #     self.W_sqrt_inv = np.linalg.inv(np.diag(self.v))
    #
    # def _cov_update(self, x_hat, sigma_y_propagated, y_hat, U):
    #     pass  # todo: currently not better than vanilla
    #     d = .1  # (1 - self.forget_factor) / (1 - self.forget_factor ** (self.timestep + 1))
    #
    #     # measurement noise
    #     # S_R_aa = chol_update(np.sqrt(1 - d) * self.S_R, np.abs(y_hat), d)
    #     # S_R_a = chol_update(S_R_aa, sigma_y_propagated-y_hat.reshape(-1, 1), -d *self.w_coeffs) # todo: ambiguous in the paper
    #     # self.S_R = np.diag(np.sqrt(np.diag(S_R_a @ S_R_a.T)))
    #
    #     # process noise
    #     S_Q_aa = chol_update(self.S_Q, np.abs(self.x_pred - x_hat), d)
    #     S_Q_a = chol_update(S_Q_aa, U, -d)
    #     self.S_Q = np.diag(np.sqrt(np.diag(S_Q_a @ S_Q_a.T)))
    #     print(self.S_Q)
    #


