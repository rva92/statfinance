from statfinance.utilities.kernels.ikernel import IKernel
from math import exp


class ExponentialKernel(IKernel):
    def __init__(self, sided: str = None):
        """
        Initializes an exponential kernel. The kernel may be right or left sided by
        passing 'right' or 'left', respectively, to the sided argument

        -----
        :param sided:
            string (defaults to None) either 'left' or 'right' for left of right sided
            exponential kernels respectively. If None, then a symmetric exponential
            kernel is used
        -----
        """
        self.sided = sided

    def evaluate(self, x: float, **kwargs) -> float:
        """
        Evaluates the kernel at a value x

        -----
        :param x:
            Float with the argument to the kernel

        :return:
            Float with the kernel value
        -----
        """
        if self.sided.lower() == 'right':
            if x >= 0:
                return exp(-x)
            else:
                return 0

        elif self.sided.lower() == 'left':
            if x <= 0:
                return exp(x)
            else:
                return 0

        else:
            return exp(x)

    @property
    def kernel_squared_constant(self):
        """
        The squared kernel value integrated over -inf to 0. This has a specific use case
        for the the DriftBurstStat and VStat

        As this is an exponential kernel (with a negative sign) it will evaluate to 1

        -----
        :return:
            Float with the squared kernel specific constant
        -----
        """
        return self.evaluate(0) - self.evaluate(-9_999_999_999)
