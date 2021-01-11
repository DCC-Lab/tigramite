"""
Parallel PCMCI: new version using Python's multiprocessing module instead of MPI, which is another dependency.
"""

import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import multiprocessing as mp
from multiprocessing import Array
import time
from scipy import sparse
import os

# Not a fan of global variables, but I don't see how I could do it otherwise. Class attribute doesn't seem to work.
__sharedMemoryVariablesDict__ = {}


class PCMCI_Parallel2:
    def __init__(self, data: np.ndarray, cond_ind_test, tau_min: int, tau_max: int, pc_alpha: float):
        self.__nbVar = data.shape[-1]
        self.__data = data
        self.__cond_ind_test = cond_ind_test
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
        self.__matricesShape = (self.__nbVar, self.__nbVar, self.__tau_max + 1)

    @staticmethod
    def split(container, count):
        container = tuple(container)
        return [container[i::count] for i in range(count)]

    def initWorker(self, pvalsMat, statValMat):
        __sharedMemoryVariablesDict__["pvals"] = pvalsMat
        __sharedMemoryVariablesDict__["statVal"] = statValMat

    def run_pc_stable_parallel_singleVariable(self, variables):
        out = []
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test())
        for variable in variables:
            start = time.time()
            parents_of_var, otherStats = pcmci_var.run_pc_stable_single_var(variable, tau_min=self.__tau_min,
                                                                            tau_max=self.__tau_max,
                                                                            pc_alpha=self.__pc_alpha,
                                                                            selected_links=self.__allSelectedLinks,
                                                                            otherReturn=False)
            print(f"PC algo done for var {variable}, time {time.time() - start} s")
            out.append([variable, pcmci_var, parents_of_var, otherStats])
        return out

    def run_mci_parallel_singleVar(self, stuff):
        out = []
        currentAllTuples = []
        processPValMatrix = np.frombuffer(__sharedMemoryVariablesDict__["pvals"].get_obj()).reshape(
            self.__matricesShape)
        processValMatrix = np.frombuffer(__sharedMemoryVariablesDict__["statVal"].get_obj()).reshape(
            self.__matricesShape)
        for variable, pcmci_var in stuff:
            currentSelectedLinks = self.__currentSelectedLinks.copy()
            currentSelectedLinks[variable] = self.__allSelectedLinks[variable]
            start = time.time()
            results_in_var = pcmci_var.run_mci(tau_min=self.__tau_min, tau_max=self.__tau_max, parents=self.all_parents,
                                               selected_links=currentSelectedLinks)
            print(f"MCI algo done for var {variable}, time {time.time() - start} s")
            processValMatrix[:, variable, :] = results_in_var["val_matrix"][:, variable, :]
            processPValMatrix[:, variable, :] = results_in_var["p_matrix"][:, variable, :]
            currentAllTuples.extend(pcmci_var.allTuples)
            out.append(variable)
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

        mci_input = []

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update(innerElem[-2])
            mci_input.append([e[:2] for e in elem])

        mci_input = self.split(mci_input, nbWorkers)

        ### Shared memory preparation ###
        sharedArraySize = np.prod(self.__matricesShape).item()
        pmatrixSharedArray = Array("d", sharedArraySize)
        valmatrixSharedArray = Array("d", sharedArraySize)
        pmatrix = np.frombuffer(pmatrixSharedArray.get_obj()).reshape(self.__matricesShape)
        valmatrix = np.frombuffer(valmatrixSharedArray.get_obj()).reshape(self.__matricesShape)
        np.copyto(pmatrix, 1)
        np.copyto(valmatrix, 0)

        start = time.time()
        initargs = (pmatrixSharedArray, valmatrixSharedArray)
        with mp.Pool(nbWorkers, initializer=self.initWorker, initargs=initargs) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, mci_input)

        print(f"MCIs done: {time.time() - start}")
        confMatrix = None

        for out in output:
            self.allTuples.extend(out[1])

        for pc_out in pc_output:
            for innerOut in pc_out:
                index = innerOut[0]
                self.val_min.update({index: innerOut[-1]["val_min"]})
                self.pval_max.update({index: innerOut[-1]["pval_max"]})

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

    path = os.path.join(os.getcwd(), "tigramite", "data", "timeSeries_ax1.npy")
    data = np.load(path).T
    data = data[:440, :10]
    seq_pcmci = PCMCI(pp.DataFrame(data), ParCorr())
    # results_pcmci_seq = seq_pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.01)
    pcmci_par2 = PCMCI_Parallel2(data, ParCorr, 0, 5, 0.01)
    start = time.time()
    results_pcmci_pa2r = pcmci_par2.start()
    print(f"Total time: {time.time() - start}")
