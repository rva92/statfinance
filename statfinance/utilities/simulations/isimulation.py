from abc import ABC, abstractmethod
from typing import List
from joblib import Parallel, delayed


class ISimulation(ABC):
    def run_simulation(self, t: int = 100, n: int = 1, n_jobs: int = 1) -> List[list]:
        """
        Simulates n paths of length t of the underlying model
        To increase speed of estimation, the n_jobs parameter can be used to simulate
        on multiple cores

        -----
        :param t:
            Integer with length of each simulation path

        :param n:
            Integer with number of paths to simulate

        :param n_jobs:
            Integer with number of cores to use. Default is 1

        :return:
            List of list with the inner list containing one simulation path
        -----
        """
        results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(self.simulate_one_path)(t=t)
            for n in range(n)
        )

        return results

    @abstractmethod
    def simulate_one_path(self, t: int = 100) -> list:
        pass
