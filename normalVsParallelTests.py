from benchmark import BenchmarkStatsFromFiles
import run_pcmci_parallel_v2
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn
import tigramite.data_processing as pp
import numpy as np
import pandas as pd
import os


class ParallelVsNormalTest:

    def __init__(self, data: np.ndarray, cond_ind_test, tau_min=0, tau_max=5, pc_alpha=0.01):
        self.__data = data
        self.__cond_ind_test = cond_ind_test
        self.__tau_min = tau_min
        self.__tau_max = tau_max
        self.__pc_alpha = pc_alpha
        self.__normalAllParents = None
        self.__parallelAllParents = None

    def runForBoth(self):
        dataframe = pp.DataFrame(self.__data)
        pcmci_norm = PCMCI(dataframe, self.__cond_ind_test)
        pcmci_par = run_pcmci_parallel_v2.PCMCI_Parallel(self.__data, self.__tau_min, self.__tau_max, self.__pc_alpha)
        print("==== Running normal PCMCI ====")
        pcmci_norm.run_pcmci(None, self.__tau_min, self.__tau_max, pc_alpha=self.__pc_alpha)
        print("==== Running parallel PCMCI ====")
        pcmci_par.start()
        self.__normalAllParents = pcmci_norm.all_parents
        self.__parallelAllParents = pcmci_par.all_parents

    def compareAllParents(self):
        if self.__normalAllParents is None or self.__parallelAllParents is None:
            raise ValueError("Nothing to compare. Please run both methods.")
        same = self.__normalAllParents == self.__parallelAllParents
        if not same:
            diff = self.dictDifference(self.__normalAllParents, self.__parallelAllParents)
            print(diff)
        return same

    @staticmethod
    def dictDifference(d1: dict, d2: dict):
        set1 = set(d1.items())
        set2 = set(d2.items())
        return set1 ^ set2


class MultipleParallelVsNormalTests:

    def __init__(self, data: list, cond_ind_test, tau_min=0, tau_max=5, pc_alpha=0.01):
        self.__data = data
        self.__cond_ind_test = cond_ind_test
        self.__tau_min = tau_min
        self.__tau_max = tau_max
        self.__pc_alpha = pc_alpha
        self.__allTestObjs = []

    def runAllForBoth(self):
        for d in self.__data:
            singleTest = ParallelVsNormalTest(d, self.__cond_ind_test, self.__tau_min, self.__tau_max, self.__pc_alpha)
            singleTest.runForBoth()
            self.__allTestObjs.append(singleTest)

    def compareAllParentsForAllTests(self):
        sames = []
        for obj in self.__allTestObjs:
            sames.append(obj.compareAllParents)
        return sames


if __name__ == '__main__':
    np.random.seed(43)


    # Example process to play around with
    # Each key refers to a variable and the incoming links are supplied
    # as a list of format [((var, -lag), coeff, function), ...]
    def lin_f(x):
        return x


    def nonlin_f(x):
        return x + 5. * x ** 2 * np.exp(-x ** 2 / 20.)


    links = {0: [((0, -1), 0.9, lin_f)],
             1: [((1, -1), 0.8, lin_f), ((0, -1), 0.8, lin_f)],
             2: [((2, -1), 0.7, lin_f), ((1, 0), 0.6, lin_f)],
             3: [((3, -1), 0.7, lin_f), ((2, 0), -0.5, lin_f)],
             }

    data, nonstat = pp.structural_causal_process(links, T=1000, seed=7)
    cond_ind_test = ParCorr()
    tau_max, tau_min = 5, 1
    pc_alpha = 0.01

    singleTest = ParallelVsNormalTest(data, cond_ind_test, tau_min, tau_max, pc_alpha)
    singleTest.runForBoth()
    singleTest.compareAllParents()
