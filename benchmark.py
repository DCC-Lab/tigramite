from tigramite.pcmci import PCMCI
import run_pcmci_parallel_v2
import time
import os
import numpy as np
import matplotlib.pyplot
from tigramite.independence_tests import ParCorr, CMIknn
import tigramite.data_processing as pp
import tigramite.plotting as tp
import pandas as pd
import warnings as warn


class Benchmark:

    def __init__(self, data: np.ndarray, parallel: bool = False, savefileNames: tuple = ("shapes.txt", "tauMax.txt")):
        self.__data = data
        self.__parallel = parallel
        self.__tauTimes = None
        self.__shapeTimes = None
        self.__columns = None
        self.__tausLines = None
        self.__shapesLines = None
        self.__saveFnames = savefileNames

    def __noBenchmarkDoneWarning(self):
        warn.warn("No benchmark done. Please run with `start` method.")

    @property
    def data(self):
        return self.__data.copy()

    @property
    def tauTimes(self):
        if self.__tauTimes is None:
            self.__noBenchmarkDoneWarning()
            return None
        return self.__tauTimes.copy()

    @property
    def shapeTimes(self):
        if self.__shapeTimes is None:
            self.__noBenchmarkDoneWarning()
            return None
        return self.__shapeTimes.copy()

    @property
    def nbRuns(self):
        if self.__columns is None:
            self.__noBenchmarkDoneWarning()
            return None
        return len(self.__columns)

    @property
    def shapesUsed(self):
        if self.__shapesLines is None:
            self.__noBenchmarkDoneWarning()
            return None
        return self.__shapesLines

    def __shapeImpact(self, shapes: list, currentRunTimes: np.ndarray, tau_max: int = 3):
        method = self.runPCMCI
        if self.__parallel:
            method = self.runPCMCI_par
        total = len(shapes)
        shapesDone = 0
        self.__shapesLines = shapes
        for shape in shapes:
            subdata = self.__data[:shape[0], :shape[1]]
            start = time.perf_counter_ns()
            method(subdata, tau_max)
            end = time.perf_counter_ns()
            currentRunTimes[shapesDone] = (end - start) / 1e9
            shapesDone += 1
            print(f"Shapes done: {shapesDone} / {total}")
            self.__saveShapeTimes()

    def __tau_maxImpact(self, tau_max_s: list, dataShape: tuple, currentRunTaus: np.ndarray):
        method = self.runPCMCI
        if self.__parallel:
            method = self.runPCMCI_par
        subdata = self.__data[:dataShape[0], :dataShape[1]]
        total = len(tau_max_s)
        tauDone = 0
        self.__tausLines = tau_max_s
        for tau in tau_max_s:
            start = time.perf_counter_ns()
            method(subdata, tau)
            end = time.perf_counter_ns()
            currentRunTaus[tauDone] = (end - start) / 1e9
            tauDone += 1
            print(f"Taus done: {tauDone} / {total}")
            self.__saveTauTimes()

    def runPCMCI(self, data: np.ndarray, tau_max: int):
        dataframe = pp.DataFrame(data)
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=0.01)

    def runPCMCI_par(self, data: np.ndarray, tau_max: int):
        dataframe = pp.DataFrame(data)
        cond_ind_test = ParCorr()
        pcmci = run_pcmci_parallel_v2.PCMCI_Parallel(data, tau_max, 0.01)
        results = pcmci.start()

    def start(self, nbRuns: int, shapes: list, taus: list, dataShape: tuple = (400, 20), tau_max: int = 3):
        self.__columns = [f"Run {i + 1}" for i in range(nbRuns)]
        nbTaus = len(taus)
        nbShapes = len(shapes)
        self.__tauTimes = np.zeros((nbTaus, nbRuns), dtype=float)
        self.__shapeTimes = np.zeros((nbShapes, nbRuns), dtype=float)
        for i, col in enumerate(self.__columns):
            print(f"{'=' * 5} Starting {col} {'=' * 5}")
            self.__shapeImpact(shapes, self.__shapeTimes[:, i], tau_max)
            self.__tau_maxImpact(taus, dataShape, self.__tauTimes[:, i])
            print(f"{'=' * 5} Ending {col} {'=' * 5}")

    def __saveTauTimes(self):
        fname = self.__saveFnames[1]
        if self.__parallel:
            fname = "par_" + fname
        df = pd.DataFrame(self.__tauTimes, index=self.__tausLines, columns=self.__columns)
        df.to_csv(fname)

    def __saveShapeTimes(self):
        fname = self.__saveFnames[0]
        if self.__parallel:
            fname = "par_" + fname
        df = pd.DataFrame(self.__shapeTimes, index=self.__shapesLines, columns=self.__columns)
        df.to_csv(fname)


class BenchmarkStats:

    def __init__(self):
        pass


if __name__ == '__main__':
    shapes = [(10, 10), (20, 10), (100, 10), (440, 10), (10, 15), (20, 15), (100, 15), (440, 15), (10, 20),
              (20, 20), (100, 20), (440, 20), (10, 25), (20, 25), (100, 25), (440, 25), (10, 30), (20, 30), (100, 30),
              (440, 30), (10, 32), (20, 32), (100, 32), (440, 32), (10, 33), (20, 33), (100, 33), (440, 33)]
    shapes2 = [(10, 10), (10, 20), (10, 25), (10, 30), (100, 10), (100, 30), (100, 20), (100, 25), (440, 25), (440, 30),
               (440, 20), (440, 10)]
    taus = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100, 150]
    datashape = (400, 20)

    path = os.path.join(os.getcwd(), "tigramite", "data", "timeSeries_ax1.npy")
    data = np.load(path).T

    b = Benchmark(data, False, ("shapes.txt", "tauMax.txt"))
    b.start(1, shapes, taus, datashape)
    b2 = Benchmark(data, False, ("shapesUnordered.txt", "tauMax.txt"))
    b2.start(1, shapes2, taus, datashape)
