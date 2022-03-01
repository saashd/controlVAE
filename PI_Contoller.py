from math import exp


def pi_controller(desired_KL, KL, beta_max, beta_min, N, Kp=0.001, Ki=-0.001):
    I_p0 = 0
    beta0 = 0
    for i in range(0, N):
        error = desired_KL - KL
        P_pi = Kp / (1.0 + float(exp(error)))
        if beta_min <= beta0 <= beta_max:
            I_pi = I_p0 - Ki * error
        else:
            I_pi = I_p0  # Anti-windup

        beta = P_pi + I_pi + beta_min

        if beta > beta_max:
            beta = beta_max
        if beta < beta_min:
            beta = beta_min

        I_p0 = I_pi
        beta0 = beta

    return beta0

class PIDControl():
    """docstring for ClassName"""

    def __init__(self):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        self.I_k1 = 0.0
        self.W_k1 = 0.0
        self.e_k1 = 0.0

    def _Kp_fun(self, Err, scale=1):
        return 1.0 / (1.0 + float(scale) * exp(Err))

    def pid(self, exp_KL, kl_loss, Kp=0.001, Ki=-0.001, Kd=0.01):
        """
        position PID algorithm
        Input: KL_loss
        return: weight for KL loss, beta
        """
        error_k = exp_KL - kl_loss
        ## comput U as the control factor
        Pk = Kp * self._Kp_fun(error_k)
        Ik = self.I_k1 - Ki * error_k
        # Dk = (error_k - self.e_k1) * Kd

        ## window up for integrator
        if self.W_k1 < 0 and self.W_k1 >= 1:
            Ik = self.I_k1

        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        self.e_k1 = error_k

        ## min and max value
        if Wk > 1:
            Wk = 1.0
        if Wk < 0:
            Wk = 0.0

        return Wk, error_k
