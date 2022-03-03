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
