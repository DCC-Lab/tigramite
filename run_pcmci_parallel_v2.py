"""
Parallel PCMCI: new version using Python's multiprocessing module instead of MPI, which is another dependency.
"""

import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import multiprocessing as mp
import time


class PCMCI_Parallel:
    def __init__(self, data: np.ndarray, tau_min: int, tau_max: int, pc_alpha: float):
        self.__nbVar = data.shape[-1]
        self.__data = data
        self.__cond_ind_test = ParCorr
        self.__tau_max = tau_max
        self.__tau_min = tau_min
        self.__pc_alpha = pc_alpha
        self.all_parents = {}
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test())
        self.__allSelectedLinks = pcmci_var._set_sel_links(None, self.__tau_min, self.__tau_max, True)
        self.__currentSelectedLinks = {key: [] for key in self.__allSelectedLinks.keys()}
        self.allTuples = []

    @staticmethod
    def split(container, count):
        container = tuple(container)
        return [container[i::count] for i in range(count)]


    def run_pc_stable_parallel_on_selected_variables(self, variables):
        """
        Method used to compute the PC_1 algorithm on selected variables. This method is run by a single process
        in order to use parallelism. For example, if one has 10 variables and 5 cpu cores, each core will run the
        PC_1 algorithm on 2 variables.

        Parameters
        ----------
        variables: An iterable
            An iterable containing a portion of the original variables. These variables will be fed to a single process
            to compute the PC algorithm on each variable.

        Returns
        -------
        out: A list
            A list containing a list for each variable used in this method. Each list has the current variable, its
            PCMCI object and its PC output.
        """
        out = []
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test())
        for variable in variables:
            start = time.time()
            parents_of_var = pcmci_var.run_pc_stable_singleVar(variable, tau_min=self.__tau_min, tau_max=self.__tau_max,
                                                               pc_alpha=self.__pc_alpha,
                                                               selected_links=self.__allSelectedLinks)
            # print(f"PC algo done for var {variable}, time {time.time() - start} s")
            out.append([variable, pcmci_var, parents_of_var])
        return out

    def run_mci_parallel_singleVar(self, stuff):
        out = []
        currentAllTuples = []
        # stuff = stuff[0]
        for variable, pcmci_var, parents_of_var in stuff:
            # print(variable)
            currentSelectedLinks = self.__currentSelectedLinks.copy()
            currentSelectedLinks[variable] = self.__allSelectedLinks[variable]
            start = time.time()
            results_in_var = pcmci_var.run_mci(tau_min=self.__tau_min, tau_max=self.__tau_max, parents=self.all_parents,
                                               selected_links=currentSelectedLinks)
            print(f"MCI algo done for var {variable}, time {time.time() - start} s")
            currentAllTuples.extend(pcmci_var.allTuples)
            out.append([variable, pcmci_var, parents_of_var, results_in_var])
        return out, currentAllTuples

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        if nbWorkers > self.__nbVar:
            nbWorkers = self.__nbVar
        splittedJobs = self.split(range(self.__nbVar), nbWorkers)

        start = time.time()
        with mp.Pool(nbWorkers) as pool:
            pc_output = pool.map(self.run_pc_stable_parallel_on_selected_variables, splittedJobs)
        # print(f"PCs done: {time.time() - start} s")

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update(innerElem[-1])
        # print(self.all_parents)
        pc_output = self.split(pc_output, nbWorkers)
        start = time.time()
        with mp.Pool(nbWorkers) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, pc_output)
        print(f"MCIs done: {time.time() - start}")
        for out in output:
            self.allTuples.extend(out[1])

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

    pcmci = PCMCI(pp.DataFrame(data), ParCorr())
    start = time.time()
    results = pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.01)
    # print(pcmci.all_parents)
    print(f"Total time: {time.time() - start}")
    pcmci_par = PCMCI_Parallel(data, 0, 5, 0.01)
    start = time.time()
    pcmci_par.start()
    # print(pcmci_par.all_parents)
    print(f"Total time: {time.time() - start}")
    print(sorted(pcmci.allTuples) == sorted(pcmci_par.allTuples))

