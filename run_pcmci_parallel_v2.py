"""
Parallel PCMCI: new version using Python's multiprocessing module instead of MPI, which is another dependency.
"""

import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import multiprocessing as mp


class PCMCI_Parallel:

    def __init__(self, data: np.ndarray, tau_min: int, tau_max: int, pc_alpha: float):
        self.__nbVar = data.shape[-1]
        self.__data = pp.DataFrame(data)
        self.__cond_ind_test = ParCorr()
        self.__tau_max = tau_max
        self.__tau_min = tau_min
        self.__pc_alpha = pc_alpha
        self.all_parents = {}

    @staticmethod
    def split(container, count):
        container = tuple(container)
        return [container[i::count] for i in range(count)]

    def run_pc_stable_parallel_singleVariable(self, variable):
        pcmci_var = PCMCI(dataframe=self.__data, cond_ind_test=self.__cond_ind_test)
        parents_of_var = pcmci_var.run_pc_stable_singleVar(variable, None, self.__tau_min, self.__tau_max,
                                                           pc_alpha=self.__pc_alpha)
        return pcmci_var, parents_of_var

    def run_PCMCI_parallel(self, variable):
        pcmci_var, parents_of_var = self.run_pc_stable_parallel_singleVariable(variable)
        results_in_var = pcmci_var.run_mci(tau_max=self.__tau_max)
        return variable, pcmci_var, parents_of_var, results_in_var

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        splittedJobs = range(self.__nbVar)

        with mp.Pool(nbWorkers) as pool:
            output = pool.map(self.run_PCMCI_parallel, splittedJobs)

        for result in output:
            currentVar = result[0]
            currentParents = result[2]
            self.all_parents[currentVar] = currentParents

        return output
