import pandas as pd


class VStat:
    def __init__(self, log_prices: pd.Series):
        self.log_prices = log_prices
        self._vstat = None

    def _estimate_v_stat(self):
        ...
