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
        parents_of_var = {variable: parents_of_var}
        return variable, pcmci_var, parents_of_var

    def run_mci_parallel_singleVar(self, variable, pcmci_var, parents_of_var):
        results_in_var = pcmci_var.run_mci(tau_max=self.__tau_max, parents=parents_of_var)
        return variable, pcmci_var, parents_of_var, results_in_var

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        splittedJobs = range(self.__nbVar)
        chunkSize = len(splittedJobs) // nbWorkers
        if chunkSize == 0:
            chunkSize = 1

        with mp.Pool(nbWorkers) as pool:
            pc_output = pool.map(self.run_pc_stable_parallel_singleVariable, splittedJobs,
                                 chunksize=chunkSize)
        with mp.Pool(nbWorkers) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, pc_output, chunksize=chunkSize)

        for result in output:
            currentParents = result[2]
            self.all_parents.update(currentParents)

        return output


class OlderVariant_PCMCI_Parallel:
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
        return variable, pcmci_var, parents_of_var

    def run_mci_parallel_singleVar(self, variable):
        variabel, pcmci_var, parents_of_var = self.run_pc_stable_parallel_singleVariable(variable)
        results_in_var = pcmci_var.run_mci(tau_max=self.__tau_max)
        return variable, pcmci_var, parents_of_var, results_in_var

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        splittedJobs = range(self.__nbVar)
        chunkSize = len(splittedJobs) // nbWorkers
        if chunkSize == 0:
            chunkSize = 1

        with mp.Pool(nbWorkers) as pool:
            output = pool.map(self.run_mci_parallel_singleVar, splittedJobs)

        for result in output:
            currentVar = result[0]
            currentParents = result[2]
            self.all_parents[currentVar] = currentParents

        return output


if __name__ == '__main__':
    np.random.seed(42)  # Fix random seed
    links_coeffs = {0: [((0, -1), 0.7)],
                    1: [((1, -1), 0.8), ((0, -1), 0.8)],
                    2: [((2, -1), 0.5), ((1, -2), 0.5)],
                    }

    T = 500  # time series length
    data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
    T, N = data.shape
    import os

    path = os.path.join(os.getcwd(), "tigramite", "data", "timeSeries_ax1.npy")
    data = np.load(path).T
    data = data[:440, :100]
    par = PCMCI_Parallel(data, 1, 5, 0.01)
    import time

    s = time.time()
    par.start()
    print(time.time() - s)
    # print(par.all_parents)
    norm = PCMCI(pp.DataFrame(data), ParCorr())
    s = time.time()
    norm.run_pcmci(tau_min=1, tau_max=5, pc_alpha=0.01)
    print(time.time() - s)
    # print(norm.all_parents)
    par_old = OlderVariant_PCMCI_Parallel(data, 1, 5, 0.01)
    s = time.time()
    par_old.start()
    print(time.time() - s)
    # print(par_old.all_parents)
    print(par.all_parents == norm.all_parents)
    print(par_old.all_parents == norm.all_parents)
    print(par_old.all_parents == par.all_parents)
