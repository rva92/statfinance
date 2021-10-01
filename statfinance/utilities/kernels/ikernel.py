from abc import ABC, abstractmethod


class IKernel(ABC):
    """
    Abstract class for kernels dictating the basic structure of all kernel
    implementations
    """
    @abstractmethod
    def evaluate(self, **kwargs) -> float:
        pass
