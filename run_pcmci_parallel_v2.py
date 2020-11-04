"""
Parallel PCMCI: new version using Python's multiprocessing module instead of MPI, which is another dependency.
"""

import numpy as np
import os, sys, pickle

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import multiprocessing as mp


class PCMCI_Parallel:

    def __init__(self, data: np.ndarray):
        self.__nbVar = data.shape[-1]
        self.__data = pp.DataFrame(data)
        self.__cond_ind_test = ParCorr()

    @staticmethod
    def split(container, count):
        return [container[i::count] for i in range(count)]

    def __run_pc_stable_parallel_singleVariable(self, variable, tau_max: int, pc_alpha: float):
        pcmci_var = PCMCI(dataframe=self.__data, cond_ind_test=self.__cond_ind_test, selected_variables=[variable])
        parents_of_var = pcmci_var.run_pc_stable(tau_max=tau_max, pc_alpha=pc_alpha)
        return pcmci_var, parents_of_var

    def __run_PCMCI_parallel(self, variable, tau_max: int, pc_alpha: float):
        pcmci_var, parents_of_var = self.__run_pc_stable_parallel_singleVariable(variable, tau_max, pc_alpha)
        results_in_var = pcmci_var.run_mci(tau_max=tau_max)
        return variable, pcmci_var, parents_of_var, results_in_var

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        splittedJobs = self.split(range(self.__nbVar), nbWorkers)
        with mp.Pool(nbWorkers) as pool:
            output = pool.map(self.__run_PCMCI_parallel, splittedJobs)

        return output
