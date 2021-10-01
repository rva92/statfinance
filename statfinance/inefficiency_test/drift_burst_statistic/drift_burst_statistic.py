import pandas as pd
from typing import Type

from statfinance.utilities.kernels.ikernel import IKernel
from statfinance.utilities.kernels.kernels import ExponentialKernel


class DriftBurstStat:
    def __init__(
            self,
            log_prices: pd.Series,
            kernel: Type[IKernel] = ExponentialKernel(sided='left'),
            bandwidth: int = 30
    ):
        """
        Implementation of the drift burst statistic based on the non parametric approach
        of Christensen, Oomen and Reno (2017)

        The statistic is based on the assumption that log prices follows a
        semi-martingale process with jumps, drift bursts and volatility bursts

        -----
        :param log_prices:
            Pandas series of log prices with DateTime index

        :param kernel:
            Instance of any IKernel subclass. Notice, for this test, it should
            be left-sided unless a forward-looking instance i desired (see VStat)

        :param bandwidth:
            Integer, with the number of observation used in the estimation of the
            local volatility. The number is equivalent to observations in
            log prices and hence the frequency dependence on the data supplied.
            Notice, the default is arbitrarily set to 30
        -----
        """
        self.log_prices = log_prices.copy()
        self.kernel = kernel
        self.bandwidth = bandwidth

    @staticmethod
    def _estimate_local_drift(
            x: pd.Series,
            bandwidth: float,
            kernel: Type[IKernel]
    ) -> float:
        """
        An estimate of a localized drift

        -----
        :param x:
            Pandas series with log prices and time as index

        :param bandwidth:
            Float with the bandwidth which should match the frequency of x

        :param kernel:
            An IKernel subtype, preferably a left-sided kernel (a right sided kernel
            has use cases wrt. VStat)

        :return:
            Float with an estimate of the localized drift
        -----
        """
        diffs_ = x.diff().dropna()
        end_time = x.index.last()
        mu = 0

        for t, log_ret in diffs_:
            mu += kernel.evaluate((t-end_time)/bandwidth) * log_ret

        mu /= bandwidth

        return mu

    @staticmethod
    def _estimate_local_volatility(
            x: pd.Series,
            bandwidth: float,
            kernel: Type[IKernel]
    ) -> float:
        """
        An estimate of a localized volatility

        -----
        :param x:
            Pandas series with log prices and time as index

        :param bandwidth:
            Float with the bandwidth which should match the frequency of x

        :param kernel:
            An IKernel subtype, preferably a left-sided kernel (a right sided kernel
            has use cases wrt. VStat)

        :return:
            Float with an estimate of the localized drift
        -----
        """
        diffs_ = x.diff().dropna().transform(lambda _: _ ** 2)
        end_time = x.index.last()
        sigma = 0

        for t, squared_diff in diffs_:
            sigma += kernel.evaluate((t - end_time) / bandwidth) * squared_diff

        sigma /= bandwidth

        return sigma ** 0.5

    def calculate_burst_stat(self, time) -> float:
        """
        Calculates the drift burst stat for a given time point

        -----
        :param time:
            Integer with the time index to calculate the drift burst stat for

        :return:
            Float with the drift burst statistic
        -----
        """
        # In case of reversed calculation, e.g. as in the forward looking part of the
        # VStat (i.e. self.bandwidth < 0), the look up order is reversed
        if self.bandwidth < 0:
            focus_prices = self.log_prices.loc[time: (time - self.bandwidth)]
        else:
            focus_prices = self.log_prices.loc[(time-self.bandwidth):time]

        mu = self._estimate_local_drift(
            x=focus_prices,
            bandwidth=self.bandwidth,
            kernel=self.kernel
        )

        sigma = self._estimate_local_volatility(
            x=focus_prices,
            bandwidth=self.bandwidth,
            kernel=self.kernel
        )

        return (self.bandwidth / self.kernel.kernel_squared_constant) ** 0.5 * mu / sigma
