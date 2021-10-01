from statfinance.utilities.simulations.isimulation import ISimulation
from numpy.random import gamma, standard_normal


class HestonModelNoDrift(ISimulation):
    def __init__(self, kappa, theta, epsilon, rho):
        """
        Simulates a Heston (1993) type stochastic volatility model with no drift

        The model has no drift and a volatility with its own path
            dX(t) = sigma(t)*dW(t)
            dsigma(t) = k(theta-sigma(t)^2)dt+eps*sigma(t)*dB(t)

        It uses a full truncation scheme to avoid potential negative variances
        The initial value of sigma is generated with a gamma distribution with
        Gamma(2*kappa*theta*epsilon**-2, 2*kappa*epsilon**-2)

        -----
        :param kappa:
            Float, ...

        :param theta:
            Float, ...

        :param epsilon:
            Float, ...

        :param rho:
            Float, ...
        -----
        """
        self.kappa = kappa
        self.theta = theta
        self.epsilon = epsilon
        self.rho = rho

    def simulate_one_path(self, t: int = 100, step_size: float = 1) -> list:
        """
        Simulates one path of length t of the Heston Model using an Euler discretisation

        -----
        :param t:
            Integer with the length of the path

        :param step_size:
            Float with the amount to increment t each time. Notice, the total time passed
            is t * step_size

        :return:
            List of floats with the path
        -----
        """
        initial_sigma = gamma(
            shape=2*self.kappa*self.theta*self.epsilon**-2,
            scale=2*self.kappa*self.epsilon**-2
        )

        sigmas = [initial_sigma]
        log_prices = []
        current_sigma = initial_sigma

        for i in range(t-1):
            current_sigma = current_sigma +\
                            self.kappa * (self.theta - current_sigma) * step_size +\
                            (self.epsilon * current_sigma) ** 0.5 * standard_normal(1)

            sigmas.append(current_sigma)
        return log_prices


# Test
model = HestonModelNoDrift(kappa=0.1, theta=0.1, epsilon=0.1, rho=0.5)
model.simulate_one_path()