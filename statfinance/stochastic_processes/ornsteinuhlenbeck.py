import numpy as np
import pandas as pd


class OrnsteinUhlenbeck:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        self.sigma_eq = None

    def fit(self, x):
        """
        Fits a stochacstic differential equation of OU form to a pandas series of data by running a simple OLS
        regression of x on a constant and its own lag. Afterwards, the OU process parameters are deducted from the OLS
        parameters

        Notice, it is assumed that x is evenly spaced in time

        -----
        :param x:
            Pandas series of flaots with the process observations
        -----
        """
        # Assumes that the time is represented in a unit scale
        delta_time = 1.

        # Get OLS estimates
        a, b, sigma_ols = self._fit_ar_model(x)

        # Convert to OU process parameters
        self.theta = -np.log(b) * 1 / delta_time
        self.mu = a / (1-b)
        self.sigma = ((sigma_ols**2 * 2 * self.theta) /(1 - b**2))**0.5
        self.sigma_eq = ((sigma_ols ** 2) / (1-b**2))**0.5

    @staticmethod
    def _fit_ar_model(x: pd.Series):
        """
        Fits an ar(1) model using OLS on a time series

        ----
        :param x:
            Pandas series of floats with the time series observations

        :return:
            Tuple with 3 floats for: the constant, the first ar-coefficient and the standard deviations of the residuals
        -----
        """
        # Create lag series and add a constant
        x_lag = x.shift(1)
        comb_data = pd.DataFrame(data={'constant': 1, 'lagged': x_lag, 'y': x}).dropna()

        xx_inv = np.linalg.inv(comb_data[['constant', 'lagged']].transpose().dot(comb_data[['constant', 'lagged']]))
        params = xx_inv.dot(comb_data[['constant', 'lagged']].transpose().dot(comb_data['y']))

        # Calculate the residuals to extract the variance of the process
        residuals = comb_data['y'] - comb_data[['constant', 'lagged']].dot(params)

        return params[0], params[1], residuals.std()


# TODO: Next project: https://hudsonthames.org/pairs-trading-based-on-renko-and-kagi-models/
# TODO: Next project: https://hudsonthames.org/pairs-trading-with-markov-regime-switching-model/

# /// Test
x = pd.Series([0.1, 0.11, 0.1, 0.13, 0.07, 0.09, 0.1])
