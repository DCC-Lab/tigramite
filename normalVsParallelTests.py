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
        self.__normalPValMax = None
        self.__parallelPValMax = None
        self.__normalValMin = None
        self.__parallelValMin = None
        self.__parallelValMatrix = None
        self.__normalValMatrix = None
        self.__parallelPValMatrix = None
        self.__normalPValMatrix = None

    def runForBoth(self):
        dataframe = pp.DataFrame(self.__data)
        pcmci_norm = PCMCI(dataframe, self.__cond_ind_test)
        pcmci_par = run_pcmci_parallel_v2.PCMCI_Parallel(self.__data, ParCorr(), self.__tau_min, self.__tau_max,
                                                         self.__pc_alpha)
        print("==== Running normal PCMCI ====")
        norm_res = pcmci_norm.run_pcmci(None, self.__tau_min, self.__tau_max, pc_alpha=self.__pc_alpha)
        print("==== Running parallel PCMCI ====")
        par_res = pcmci_par.start()
        self.__normalAllParents = pcmci_norm.all_parents
        self.__parallelAllParents = pcmci_par.all_parents
        self.__normalPValMax = pcmci_norm.pval_max
        self.__parallelPValMax = pcmci_par.pval_max
        self.__normalValMin = pcmci_norm.val_min
        self.__parallelValMin = pcmci_par.val_min
        self.__normalValMatrix = norm_res["val_matrix"]
        self.__parallelValMatrix = par_res["val_matrix"]
        self.__normalPValMatrix = norm_res["p_matrix"]
        self.__parallelPValMatrix = par_res["p_matrix"]

    def compareAllParentsAndMCITuples(self, saveDiffToFile: bool = True, diffFilename: str = None):
        if self.__normalAllParents is None or self.__parallelAllParents is None:
            raise ValueError("Nothing to compare. Please run both methods.")
        sameParents = self.__normalAllParents == self.__parallelAllParents
        samePValMax = self.__normalPValMax == self.__parallelPValMax
        sameValMin = self.__normalValMin == self.__parallelValMin
        sameMCIPValMatrix = np.array_equal(self.__parallelPValMatrix, self.__normalPValMatrix)
        sameMCIValMatrix = np.array_equal(self.__parallelValMatrix, self.__normalValMatrix)
        diff = ""
        if not sameParents:
            diff += f"Sequential parents:\n{self.__normalAllParents}\nParallel parents:\n{self.__parallelAllParents}"
        if not samePValMax:
            diff += f"Sequential max pvals:\n{self.__normalPValMax}\nParallel max pvals:\n{self.__normalPValMax}"
        if not sameValMin:
            diff += f"Sequential min vals:\n{self.__normalValMin}\nParallel min vals:\n{self.__parallelValMin}"
        if not sameMCIPValMatrix:
            diff += f"Sequential MCI pvals:\n{self.__normalPValMatrix}\nParallel min pvals:\n{self.__parallelPValMatrix}"
        if not sameMCIValMatrix:
            diff += f"Sequential MCI vals:\n{self.__normalValMatrix}\nParallel min vals:\n{self.__parallelValMatrix}"
        if diff != "":
            if saveDiffToFile:
                if diffFilename is None:
                    diffFilename = "differences.txt"
                if not diffFilename.endswith(".txt"):
                    diffFilename += ".txt"
                with open(diffFilename, "w") as f:
                    f.write(diff)
            print(diff)
        return sameParents and samePValMax


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

    def compareForAllTests(self, saveDiffToFile: bool = True):
        sames = []
        for i, obj in enumerate(self.__allTestObjs):
            diffFilename = f"data{i + 1}.txt"
            sames.append(obj.compareAllParentsAndMCITuples(saveDiffToFile, diffFilename))
        print(sames)
        return sames


if __name__ == '__main__':
    cond_ind_test = ParCorr()
    tau_max, tau_min = 5, 0
    pc_alpha = 0.01

    path = os.path.join(os.getcwd(), "tigramite", "data", "timeSeries_ax1.npy")
    data = np.load(path).T
    shapes = [(50, 10), (100, 10), (200, 10), (300, 10), (400, 10), (440, 10), (50, 20), (100, 20), (200, 20),
              (300, 20), (400, 20), (440, 20), (50, 40), (100, 40), (200, 40), (300, 40), (400, 40), (440, 40),
              (440, 400)]
    allData = [data[:shape[0], :shape[1]] for shape in shapes]
    m = MultipleParallelVsNormalTests(allData, cond_ind_test, tau_min, tau_max, pc_alpha)
    m.runAllForBoth()
    m.compareForAllTests()
