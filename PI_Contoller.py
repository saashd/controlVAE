from math import exp


def pi_controller(desired_KL, KL, beta_max, beta_min, N, Kp=0.001, Ki=-0.001):
    I_pi = 0
    beta = 0
    for t in range(1, N):
        error = desired_KL - KL
        P_pi = Kp / (1 + exp(error))
        if beta_min <= beta <= beta_max:
            I_pi = I_pi - Ki * error
        else:
            I_pi = I_pi  # Anti-windup

        beta = P_pi + I_pi + beta_min

        if beta > beta_max:
            beta = beta_max
        if beta < beta_min:
            beta = beta_min
    return beta
