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
    def __init__(self, data: np.ndarray, cond_indep_test: object = ParCorr(), tau_min: int = 0, tau_max: int = 5,
                 pc_alpha: float = 0.01):
        self.__nbVar = data.shape[-1]
        self.__data = data
        self.__cond_ind_test = cond_indep_test
        self.__tau_max = tau_max
        self.__tau_min = tau_min
        self.__pc_alpha = pc_alpha
        self.all_parents = {}
        self.val_min = {}
        self.pval_max = {}
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test)
        self.__allSelectedLinks = pcmci_var._set_sel_links(None, self.__tau_min, self.__tau_max, False)
        self.__currentSelectedLinks = {key: [] for key in self.__allSelectedLinks.keys()}
        self.allTuples = []

    @staticmethod
    def split(container, count):
        container = tuple(container)
        return [container[i::count] for i in range(count)]

    def run_pc_stable_parallel_single_variable(self, variables):
        out = []
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test)
        for variable in variables:
            start = time.time()
            results = pcmci_var.run_pc_stable_single_var(variable, tau_min=self.__tau_min, tau_max=self.__tau_max,
                                                         pc_alpha=self.__pc_alpha,
                                                         selected_links=self.__allSelectedLinks)
            # print(f"PC algo done for var {variable}, time {time.time() - start} s")
            out.append([variable, pcmci_var, results])
        return out

    def run_mci_parallel_singleVar(self, stuff):
        out = []
        currentAllTuples = []
        # stuff = stuff[0]
        for variable, pcmci_var, pc_output in stuff:
            currentSelectedLinks = self.__currentSelectedLinks.copy()
            currentSelectedLinks[variable] = self.__allSelectedLinks[variable]
            results_in_var = pcmci_var.run_mci(tau_min=self.__tau_min, tau_max=self.__tau_max, parents=self.all_parents,
                                               selected_links=currentSelectedLinks)
            currentAllTuples.extend(pcmci_var.allTuples)
            out.append([variable, pcmci_var, pc_output, results_in_var])
        return out, currentAllTuples

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        if nbWorkers > self.__nbVar:
            nbWorkers = self.__nbVar
        splittedJobs = self.split(range(self.__nbVar), nbWorkers)

        with mp.Pool(nbWorkers) as pool:
            pc_output = pool.map(self.run_pc_stable_parallel_single_variable, splittedJobs)

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update({innerElem[0]: innerElem[-1]["parents"]})
                self.val_min.update({innerElem[0]: innerElem[-1]["val_min"]})
                self.pval_max.update({innerElem[0]: innerElem[-1]["pval_max"]})

        pc_output = self.split(pc_output, nbWorkers)

        with mp.Pool(nbWorkers) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, pc_output)
        pmatrix = np.ones((self.__nbVar, self.__nbVar, self.__tau_max + 1))
        valmatrix = pmatrix.copy()
        confMatrix = None
        for out in output:
            self.allTuples.extend(out[1])
            for index, innerOut in enumerate(out[0]):
                if index == 0:
                    print(innerOut[-1]["p_matrix"])
                pmatrix[:, index, :] = innerOut[-1]["p_matrix"][:, index, :]
                valmatrix[:, index, :] = innerOut[-1]["val_matrix"][:, index, :]
                # print(innerOut[-1]["p_matrix"])
        return {"val_matrix": valmatrix, "p_matrix": pmatrix}


class PCMCI_Parallel2:
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
        self.temp = pcmci_var._set_sel_links(self.__allSelectedLinks, 1, self.__tau_max, True)
        self.__currentSelectedLinks = {key: [] for key in self.__allSelectedLinks.keys()}
        self.allTuples = []

    @staticmethod
    def split(container, count):
        container = tuple(container)
        return [container[i::count] for i in range(count)]

    def run_pc_stable_parallel_singleVariable(self, variables):
        out = []
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test())
        for variable in variables:
            start = time.time()
            parents_of_var = pcmci_var._run_pc_stable_single(variable, tau_min=self.__tau_min, tau_max=self.__tau_max,
                                                             pc_alpha=self.__pc_alpha,
                                                             selected_links=self.temp[variable])
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
            # print(f"MCI algo done for var {variable}, time {time.time() - start} s")
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
            pc_output = pool.map(self.run_pc_stable_parallel_singleVariable, splittedJobs)
        # print(f"PCs done: {time.time() - start} s")

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update({innerElem[0]: innerElem[-1]["parents"]})
        # print(self.all_parents)
        pc_output = self.split(pc_output, nbWorkers)
        start = time.time()
        with mp.Pool(nbWorkers) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, pc_output)
        print(f"MCIs done: {time.time() - start}")
        for out in output:
            self.allTuples.extend(out[1])
            for inner in out:
                print(inner)
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
    data = data[:440, :10]

    pcmci = PCMCI(pp.DataFrame(data), ParCorr())
    start = time.time()
    results_pcmci = pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.01)
    # print(pcmci.all_parents)
    print(f"Total time: {time.time() - start}")
    pcmci_par = PCMCI_Parallel(data, ParCorr(), 0, 5, 0.01)
    start = time.time()
    results_pcmci_par = pcmci_par.start()
    # print(pcmci_par.all_parents)
    print(f"Total time: {time.time() - start}")
    print(sorted(pcmci.allTuples) == sorted(pcmci_par.allTuples))

    print(pcmci_par.all_parents == pcmci.all_parents)
    print(pcmci_par.val_min == pcmci.val_min)
    print(pcmci_par.pval_max == pcmci.pval_max)
    val_matrix_normal = results_pcmci["val_matrix"]
    val_matrix_par = results_pcmci["val_matrix"]
    pval_matrix_normal = results_pcmci["p_matrix"]
    pval_matrix_par = results_pcmci["p_matrix"]
    print(np.array_equal(val_matrix_normal, val_matrix_par))
    print(np.array_equal(pval_matrix_normal, pval_matrix_par))
    # print(sorted(pcmci.all_parents.items()))
    # print(sorted(pcmci_par2.all_parents.items()))

    # print(pcmci.all_parents)
    # p_val = results["p_matrix"]
    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(2, 3)
    # currSlice = 0
    # for row in range(2):
    #     for col in range(3):
    #         axes[row, col % 3].imshow(p_val[:, :, currSlice])
    #         axes[row, col % 3].set_title(rf"$\tau$ = {-currSlice}")
    #         currSlice += 1
    # plt.show()
