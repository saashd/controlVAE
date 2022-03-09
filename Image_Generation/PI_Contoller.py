from math import exp


class PI_Controller():
    def __init__(self):
        self.I_p = 0
        self.beta = 0
        self.error = 0

    def pi(self, desired_KL, KL, beta_max=1, beta_min=0, N=1, Kp=0.01, Ki=0.0001):
        beta_i = None
        for i in range(0, N):
            error_i = desired_KL - KL
            P_i = Kp / (1.0 + exp(error_i))
            I_i = self.I_p - Ki * error_i
            if 0 > self.beta >= 1:
                I_i = self.I_p  # Anti-windup
            beta_i = P_i + I_i + beta_min

            self.I_p = I_i
            self.beta = beta_i
            self.error = error_i

            if beta_i > beta_max:
                beta_i = beta_max
            if beta_i < beta_min:
                beta_i = beta_min

        return beta_i
