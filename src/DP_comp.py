# From PE code base https://github.com/microsoft/DPSDA/blob/main/pe/dp/gaussian.py
from scipy.optimize import root_scalar
from scipy.stats import norm
import numpy as np
def delta_Gaussian(eps, mu):
    """Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu.

    :param eps: The epsilon value
    :type eps: float
    :param mu: The mu value
    :type mu: float
    :return: The delta value
    :rtype: float
    """
    if mu == 0:
        return 0
    # Avoid overflow warnings from exp(eps) at very large eps.
    if eps > np.log(np.finfo(float).max):
        return 0
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_Gaussian(delta, mu, max_epsilon):
    """Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu.

    :param delta: The delta value
    :type delta: float
    :param mu: The mu value
    :type mu: float
    :param max_epsilon: The maximum epsilon value to search for
    :type max_epsilon: float
    """

    def f(x):
        return delta_Gaussian(x, mu) - delta

    a = 0.0
    b = float(max_epsilon)
    if b <= a:
        raise ValueError("Require max_epsilon > 0")

    fa = f(a)
    if fa <= 0:
        return 0.0
    fb = f(b)
    # If upper endpoint is still positive, expand until we cross zero.
    if fb > 0:
        for _ in range(60):
            b *= 2
            fb = f(b)
            if fb <= 0:
                break
        else:
            raise ValueError(
                "Failed to bracket epsilon root: increase max_epsilon or check delta."
            )

    return root_scalar(f, bracket=[a, b], method="brentq").root


def compute_epsilon(noise_multiplier, num_steps, delta, max_epsilon=1e7):
    """Compute epsilon of Gaussian mechanism.

    :param noise_multiplier: The noise multiplier
    :type noise_multiplier: float
    :param num_steps: The number of steps
    :type num_steps: int
    :param delta: The delta value
    :type delta: float
    :param max_epsilon: The maximum epsilon value to search for, defaults to 1e7
    :type max_epsilon: float, optional
    :return: The epsilon value.
    :rtype: float
    """
    if noise_multiplier == 0:
        print("Since noise_multiplier is 0, epsilon is INF.")
        return np.inf
    return eps_Gaussian(delta=delta, mu=np.sqrt(num_steps) / noise_multiplier, max_epsilon=max_epsilon)


def get_noise_multiplier(
    epsilon,
    num_steps,
    delta,
    min_noise_multiplier=1e-1,
    max_noise_multiplier=500,
    max_epsilon=1e7,
):
    """Get noise multiplier of Gaussian mechanism.

    :param epsilon: The epsilon value
    :type epsilon: float
    :param num_steps: The number of steps
    :type num_steps: int
    :param delta: The delta value
    :type delta: float
    :param min_noise_multiplier: The minimum noise multiplier to search for, defaults to 1e-1
    :type min_noise_multiplier: float, optional
    :param max_noise_multiplier: The maximum noise multiplier to search for, defaults to 500
    :type max_noise_multiplier: float, optional
    :param max_epsilon: The maximum epsilon value to search for, defaults to 1e7
    :type max_epsilon: float, optional
    """

    if epsilon == np.inf:
        return 0.0

    def objective(x):
        return (
            compute_epsilon(
                noise_multiplier=x,
                num_steps=num_steps,
                delta=delta,
                max_epsilon=max_epsilon,
            )
            - epsilon
        )

    output = root_scalar(objective, bracket=[min_noise_multiplier, max_noise_multiplier], method="brentq")

    if not output.converged:
        raise ValueError("Failed to converge")
    if output.root <= 0:
        raise ValueError("The noise multiplier <= 0, which may not provide any privacy guarantee.")

    return output.root