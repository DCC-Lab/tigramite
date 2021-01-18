"""
Parallel PCMCI: new version using Python's multiprocessing module instead of MPI, which is another dependency.
"""

import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from multiprocessing import Array, RawArray, Manager, Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import time
from scipy import sparse
import os

# Not a fan of global variables, but I don't see how I could do it otherwise. Class attribute doesn't seem to work.
__sharedMemoryVariablesDict__ = {}


class PCMCI_Parallel:
    def __init__(self, data: np.ndarray, cond_ind_test: callable, tau_min: int, tau_max: int, pc_alpha: float):
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
        self.__matricesShape = (self.__nbVar, self.__nbVar, self.__tau_max + 1)
        self.__sharedAllParents = Manager().dict()

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
        processPValMatrix = np.frombuffer(__sharedMemoryVariablesDict__["pvals"]).reshape(
            self.__matricesShape)
        processValMatrix = np.frombuffer(__sharedMemoryVariablesDict__["statVal"]).reshape(
            self.__matricesShape)
        for variable, pcmci_var in stuff:
            currentSelectedLinks = self.__currentSelectedLinks.copy()
            currentSelectedLinks[variable] = self.__allSelectedLinks[variable]
            start = time.time()
            results_in_var = pcmci_var.run_mci(tau_min=self.__tau_min, tau_max=self.__tau_max,
                                               parents=self.__sharedAllParents, selected_links=currentSelectedLinks)
            print(f"MCI algo done for var {variable}, time {time.time() - start} s")
            processValMatrix[:, variable, :] = results_in_var["val_matrix"][:, variable, :]
            processPValMatrix[:, variable, :] = results_in_var["p_matrix"][:, variable, :]
            out.append(variable)
        return out

    def start(self, nbWorkers: int = None):

        if nbWorkers is None:
            nbWorkers = cpu_count()
        if nbWorkers > cpu_count():
            nbWorkers = cpu_count()

        if nbWorkers > self.__nbVar:
            nbWorkers = self.__nbVar
        splittedJobs = self.split(range(self.__nbVar), nbWorkers)
        start = time.time()
        with Pool(nbWorkers) as pool:
            pc_output = pool.map(self.run_pc_stable_parallel_singleVariable, splittedJobs)
        print(f"PCs done: {time.time() - start} s")

        mci_input = []

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update(innerElem[-2])
                self.__sharedAllParents.update(innerElem[-2])
            mci_input.append([e[:2] for e in elem])

        mci_input = self.split(mci_input, nbWorkers)

        ### Shared memory preparation ###
        sharedArraySize = np.prod(self.__matricesShape).item()
        pmatrixSharedArray = RawArray("d", sharedArraySize)
        valmatrixSharedArray = RawArray("d", sharedArraySize)
        pmatrix = np.frombuffer(pmatrixSharedArray).reshape(self.__matricesShape)
        valmatrix = np.frombuffer(valmatrixSharedArray).reshape(self.__matricesShape)
        np.copyto(pmatrix, 1)
        np.copyto(valmatrix, 0)

        start = time.time()
        initargs = (pmatrixSharedArray, valmatrixSharedArray)
        with Pool(nbWorkers, initializer=self.initWorker, initargs=initargs) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, mci_input)

        print(f"MCIs done: {time.time() - start}")
        confMatrix = None

        for pc_out in pc_output:
            for innerOut in pc_out:
                index = innerOut[0]
                self.val_min.update({index: innerOut[-1]["val_min"]})
                self.pval_max.update({index: innerOut[-1]["pval_max"]})

        return {"val_matrix": valmatrix, "p_matrix": pmatrix}


class PCMCI_Parallel2:

    def __init__(self, data: np.ndarray, cond_ind_test: callable, tau_min: int, tau_max: int, pc_alpha: float):
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
        self.__matricesShape = (self.__nbVar, self.__nbVar, self.__tau_max + 1)
        self.__sharedMemoryPValName = None
        self.__sharedMemoryValName = None
        self.__sharedAllParents = Manager().dict()
        self.valmatrix, self.pmatrix = None, None

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
                                                                            otherReturn=False)
            print(f"PC algo done for var {variable}, time {time.time() - start} s")
            out.append([variable, pcmci_var, parents_of_var, otherStats])
        return out

    def run_mci_parallel_singleVar(self, stuff):
        out = []
        pvalShared = SharedMemory(self.__sharedMemoryPValName)
        valShared = SharedMemory(self.__sharedMemoryValName)
        processPValMatrix = np.ndarray(self.__matricesShape, dtype=float, buffer=pvalShared.buf)
        processValMatrix = np.ndarray(self.__matricesShape, dtype=float, buffer=valShared.buf)
        for variable, pcmci_var in stuff:
            currentSelectedLinks = self.__currentSelectedLinks.copy()
            currentSelectedLinks[variable] = self.__allSelectedLinks[variable]
            start = time.time()
            results_in_var = pcmci_var.run_mci(tau_min=self.__tau_min, tau_max=self.__tau_max,
                                               parents=self.__sharedAllParents, selected_links=currentSelectedLinks)
            print(f"MCI algo done for var {variable}, time {time.time() - start} s")
            processValMatrix[:, variable, :] = results_in_var["val_matrix"][:, variable, :]
            processPValMatrix[:, variable, :] = results_in_var["p_matrix"][:, variable, :]
            out.append(variable)
        pvalShared.close()
        valShared.close()
        return out

    def start(self, nbWorkers: int = None):
        print("Initializing multiprocessing start...")
        if nbWorkers is None:
            nbWorkers = cpu_count()
        if nbWorkers > cpu_count():
            nbWorkers = cpu_count()

        if nbWorkers > self.__nbVar:
            nbWorkers = self.__nbVar
        splittedJobs = self.split(range(self.__nbVar), nbWorkers)
        start = time.time()
        print("Starting PC step...")
        with Pool(nbWorkers) as pool:
            pc_output = pool.map(self.run_pc_stable_parallel_singleVariable, splittedJobs)
        print(f"PCs done: {time.time() - start} s")

        mci_input = []

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update(innerElem[-2])
                self.__sharedAllParents.update(innerElem[-2])
            mci_input.append([e[:2] for e in elem])

        mci_input = self.split(mci_input, nbWorkers)

        ### Shared memory preparation ###
        temp_pmatrix = np.ones(self.__matricesShape)
        temp_valmatrix = np.zeros(self.__matricesShape)
        print("Initializing memory share...")
        with SharedMemoryManager() as manager:
            sharedMemoryPmatrix = manager.SharedMemory(temp_pmatrix.nbytes)
            self.__sharedMemoryPValName = sharedMemoryPmatrix.name
            sharedMemoryValmatrix = manager.SharedMemory(temp_valmatrix.nbytes)
            self.__sharedMemoryValName = sharedMemoryValmatrix.name
            pmatrix = np.ndarray(self.__matricesShape, dtype=float, buffer=sharedMemoryPmatrix.buf)
            valmatrix = np.ndarray(self.__matricesShape, dtype=float, buffer=sharedMemoryValmatrix.buf)
            np.copyto(pmatrix, temp_pmatrix)
            np.copyto(valmatrix, temp_valmatrix)

            start = time.time()
            print("Starting MCI step...")
            with Pool(nbWorkers) as pool:
                output = pool.starmap(self.run_mci_parallel_singleVar, mci_input)
            self.pmatrix = np.copy(pmatrix)
            self.valmatrix = np.copy(valmatrix)

        print(f"MCIs done: {time.time() - start}")
        confMatrix = None

        for pc_out in pc_output:
            for innerOut in pc_out:
                index = innerOut[0]
                self.val_min.update({index: innerOut[-1]["val_min"]})
                self.pval_max.update({index: innerOut[-1]["pval_max"]})

        return {"val_matrix": self.valmatrix, "p_matrix": self.pmatrix}

    def significantLinksFound(self, p_valueThreshold: float = 0.01, *args, **kwargs):
        if self.pmatrix is None:
            raise ValueError("Please run the PCMCI algorithm (with `start` method).")
        return self.significantLinks(self.pmatrix, p_valueThreshold, *args, **kwargs)

    def significantLinksComparison(self, trueAdjacencyMatrix: np.ndarray, p_valueThreshold: float = 0.01,
                                   printReturnMessage: bool = True, *args, **kwargs):

        currentLinks = self.significantLinksFound(p_valueThreshold, *args, **kwargs)
        info = {}
        trueLinks = self.significantLinks(trueAdjacencyMatrix, p_valueThreshold, *args, **kwargs)
        truePositives = currentLinks.intersection(trueLinks)
        falsePositives = currentLinks - trueLinks
        falseNegatives = trueLinks - currentLinks
        info["true positives"] = truePositives
        info["false positives"] = falsePositives
        info["false negatives"] = falseNegatives
        if printReturnMessage:
            msg = "Info: dictionary of 3 elements. The first is the number of true positives, i.e. " \
                  "the number of links that are both true and found by the algorithm. The second is the number of " \
                  "false positives, i.e. the number of links that are found by the algorithm, but not true. " \
                  "The third is the number of false negatives, i.e. the number of links that are true, but not " \
                  "found by the algorithm."
            msg += "\nReturns also the number of true links and the number of links found."
            print(msg)
        return info, len(trueLinks), len(currentLinks)

    @staticmethod
    def significantLinks(array: np.ndarray, p_valueThreshold: float = 0.01, i_to_j: bool = True,
                         showPrints: bool = True):
        t_array = array
        if not isinstance(t_array, np.ndarray):
            if showPrints:
                print("`array` is not a NumPy ndarray. Trying conversion.")
            t_array = np.array(t_array)
        if t_array.ndim >= 3 and showPrints:
            print("We only keep the first two indices when working with array with more than 2 dimensions.")
        indicesWhereInferior = np.where(t_array <= p_valueThreshold)
        if i_to_j:
            indicesWhereInferior = {(indicesWhereInferior[0][elem], indicesWhereInferior[1][elem]) for elem in
                                    range(len(indicesWhereInferior[0]))}
        else:
            indicesWhereInferior = {(indicesWhereInferior[1][elem], indicesWhereInferior[0][elem]) for elem in
                                    range(len(indicesWhereInferior[0]))}
        return indicesWhereInferior


if __name__ == '__main__':
    a = np.array([[1, 1e-6], [1, 1], [1, 1]])
    b = np.array([[1, 1e-6], [1, 1], [1, 1]])
    a = np.dstack([a, b])
    s = PCMCI_Parallel2.significantLinks(a, 0.1, False, False)
    a_links = {(1, 0)}
    print(a_links == s)
    print(s)
    exit()
    path = os.path.join(os.getcwd(), "data", "timeSeries_ax1.npy")
    data = np.load(path).T
    data = data[:440, :10]
    seq_pcmci = PCMCI(pp.DataFrame(data), ParCorr())
    # results_pcmci_seq = seq_pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.01)
    pcmci_par = PCMCI_Parallel(data, ParCorr, 0, 5, 0.01)
    start = time.time()
    results_pcmci_par = pcmci_par.start()
    print(f"Total time: {time.time() - start}")
    # exit()
    pcmci_par2 = PCMCI_Parallel2(data, ParCorr, 0, 5, 0.01)
    start = time.time()
    results_pcmci_par2 = pcmci_par2.start()
    print(f"Total time: {time.time() - start}")
    print(np.allclose(results_pcmci_par["val_matrix"], results_pcmci_par2["val_matrix"]))
    # print(results_pcmci_par["val_matrix"])
    # print(results_pcmci_par2["val_matrix"])
    print(np.allclose(results_pcmci_par["p_matrix"], results_pcmci_par2["p_matrix"]))
    print(pcmci_par.all_parents == pcmci_par2.all_parents)
