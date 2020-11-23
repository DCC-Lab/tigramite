"""
Parallel PCMCI: new version using Python's multiprocessing module instead of MPI, which is another dependency.
"""

import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import multiprocessing as mp
import time
from copy import deepcopy


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
            pcstart = time.time()
            results = pcmci_var.run_pc_stable_single_var(variable, tau_min=self.__tau_min, tau_max=self.__tau_max,
                                                         pc_alpha=self.__pc_alpha,
                                                         selected_links=self.__allSelectedLinks)
            print(f"PC algo done for var {variable}, time {time.time() - pcstart} s")
            out.append([variable, pcmci_var, results])
        return out

    def run_mci_parallel_singleVar(self, *inputs):
        out = []
        currentAllTuples = []
        for variable, pcmci_var, pc_output in inputs:
            mciStart = time.time()
            currentSelectedLinks = self.__currentSelectedLinks.copy()
            currentSelectedLinks[variable] = self.__allSelectedLinks[variable]
            results_in_var = pcmci_var.run_mci(tau_min=self.__tau_min, tau_max=self.__tau_max, parents=self.all_parents,
                                               selected_links=currentSelectedLinks)
            out.append([variable, pcmci_var, pc_output, results_in_var])
            currentAllTuples.extend(pcmci_var.allTuples)
            print(f"MCI done for var {variable} : {time.time() - mciStart}s")
        return out, currentAllTuples

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        if nbWorkers > self.__nbVar:
            nbWorkers = self.__nbVar

        splittedJobs = self.split(range(self.__nbVar), nbWorkers)

        allPCs = time.time()
        with mp.Pool(nbWorkers) as pool:
            pc_output = pool.map(self.run_pc_stable_parallel_single_variable, splittedJobs)
        print(f"All PCs done : {time.time() - allPCs}")

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update({innerElem[0]: innerElem[-1]["parents"]})
                self.val_min.update({innerElem[0]: innerElem[-1]["val_min"]})
                self.pval_max.update({innerElem[0]: innerElem[-1]["pval_max"]})

        allMCIs = time.time()
        with mp.Pool(nbWorkers) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, pc_output)
        print(f"All MCIs done : {time.time() - allMCIs}")

        pmatrix = np.ones((self.__nbVar, self.__nbVar, self.__tau_max + 1))
        valmatrix = pmatrix.copy()
        confMatrix = None
        for out in output:
            # print(out)
            self.allTuples.extend(out[1])
            for innerOut in out[0]:
                index = innerOut[0]
                pmatrix[:, index, :] = innerOut[-1]["p_matrix"][:, index, :]
                valmatrix[:, index, :] = innerOut[-1]["val_matrix"][:, index, :]
                # print(innerOut[-1]["p_matrix"][:, index, :])

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
        self.val_min = {}
        self.pval_max = {}
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test())
        self.__allSelectedLinks = pcmci_var._set_sel_links(None, self.__tau_min, self.__tau_max, True)
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
            parents_of_var, otherStats = pcmci_var.run_pc_stable_single_var(variable, tau_min=self.__tau_min,
                                                                            tau_max=self.__tau_max,
                                                                            pc_alpha=self.__pc_alpha,
                                                                            selected_links=self.__allSelectedLinks,
                                                                            parentsOnly=False)
            print(f"PC algo done for var {variable}, time {time.time() - start} s")
            out.append([variable, pcmci_var, parents_of_var, otherStats])
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
            pc_output = pool.map(self.run_pc_stable_parallel_singleVariable, splittedJobs)
        print(f"PCs done: {time.time() - start} s")
        originalPcOutput = deepcopy(pc_output)

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update(innerElem[-2])
                # self.val_min.update({innerElem[0]: innerElem[-1]["val_min"]})
                # self.pval_max.update({innerElem[0]: innerElem[-1]["pval_max"]})
        # print(self.all_parents)
        pc_output = self.split(pc_output, nbWorkers)
        start = time.time()
        with mp.Pool(nbWorkers) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, pc_output)
        print(f"MCIs done: {time.time() - start}")
        pmatrix = np.ones((self.__nbVar, self.__nbVar, self.__tau_max + 1))
        valmatrix = pmatrix.copy()
        confMatrix = None
        for out in output:
            # print(out)
            self.allTuples.extend(out[1])
            for innerOut in out[0]:
                index = innerOut[0]
                pmatrix[:, index, :] = innerOut[-1]["p_matrix"][:, index, :]
                valmatrix[:, index, :] = innerOut[-1]["val_matrix"][:, index, :]
                # print(innerOut[-1]["p_matrix"][:, index, :])

        return {"val_matrix": valmatrix, "p_matrix": pmatrix}


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
    seq_pcmci = PCMCI(pp.DataFrame(data), ParCorr())
    results_pcmci_seq = seq_pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.01)
    pcmci_par = PCMCI_Parallel(data, ParCorr(), 0, 5, 0.01)
    start = time.time()
    results_pcmci_par = pcmci_par.start()
    # print(pcmci_par.all_parents)
    print(f"Total time: {time.time() - start}")
    pcmci_par2 = PCMCI_Parallel2(data, 0, 5, 0.01)
    start = time.time()
    results_pcmci_pa2r = pcmci_par2.start()
    # print(pcmci_par.all_parents)
    print(f"Total time: {time.time() - start}")
    print("Parents: ", seq_pcmci.all_parents == pcmci_par.all_parents)
    print("All tuples: ", sorted(seq_pcmci.allTuples) == sorted(pcmci_par.allTuples))
    print("Vals min: ", seq_pcmci.val_min == pcmci_par.val_min)
    print("p vals max: ", seq_pcmci.pval_max == pcmci_par.pval_max)
    print("MCI vals : ", np.allclose(results_pcmci_seq["val_matrix"], results_pcmci_par["val_matrix"], 1e-10, 1e-10))
    print("MCI p vals : ", np.allclose(results_pcmci_seq["p_matrix"], results_pcmci_par["p_matrix"], 1e-10, 1e-10))
    # print(results_pcmci_seq["p_matrix"])
    # print(results_pcmci_par["p_matrix"])
    print(results_pcmci_seq["p_matrix"] - results_pcmci_par["p_matrix"])
