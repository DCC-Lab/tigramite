from tigramite.pcmci import PCMCI
import run_pcmci_parallel_v2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tigramite.independence_tests import ParCorr, CMIknn
import tigramite.data_processing as pp
import pandas as pd
import warnings as warn
import typing
import re


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
        self.__shapesDataframe = None
        self.__tausDataframe = None

    def __noBenchmarkDoneWarning(self):
        warn.warn("No benchmark done. Please run with `start` method.")

    @property
    def completDataframes(self):
        if self.__shapesDataframe is None or self.__tausDataframe is None:
            self.__noBenchmarkDoneWarning()
            return None
        return self.__shapesDataframe, self.__tausDataframe

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
        self.__tausDataframe = df

    def __saveShapeTimes(self):
        fname = self.__saveFnames[0]
        if self.__parallel:
            fname = "par_" + fname
        df = pd.DataFrame(self.__shapeTimes, index=self.__shapesLines, columns=self.__columns)
        df.to_csv(fname)
        self.__shapesDataframe = df


class BaseBenchmarkStats:

    def __init__(self, shapesTimeDataframe: pd.DataFrame, tausTimeDataframe: pd.DataFrame):
        self.__dataframeShapes = shapesTimeDataframe
        self.__dataframeTaus = tausTimeDataframe
        # Cannot put both in the same stack, they might not have the same shape!
        self.__rowiseMeanShapes = self.computeRowiseMean(self.__dataframeShapes)
        self.__rowiseMeanTaus = self.computeRowiseMean(self.__dataframeTaus)
        self.__rowiseStdDevShapes = self.computeRowiseStandardDeviation(self.__dataframeShapes)
        self.__rowiseStdDevTaus = self.computeRowiseStandardDeviation(self.__dataframeTaus)

    @property
    def shapeTimeMean(self):
        return self.__rowiseMeanShapes

    @property
    def tauTimeMean(self):
        return self.__rowiseMeanTaus

    @property
    def shapeTimeStdDev(self):
        return self.__rowiseStdDevShapes

    @property
    def tauTimeStdDev(self):
        return self.__rowiseStdDevTaus

    @property
    def tausDataframe(self):
        return self.__dataframeTaus

    @property
    def shapesDataframe(self):
        return self.__dataframeShapes

    def plotTimeFctShape(self, show: bool = True, savefig: bool = True, figname: str = None):
        index = self.__dataframeShapes.index
        nbRuns = len(self.__dataframeShapes.columns) - 1
        # Retrieve tuples from strings
        if any(not isinstance(s, tuple) for s in index):
            index = [tuple(map(int, s[1: -1].split(","))) for s in index]
        nbVars = [i[-1] for i in index]
        nbTimeSteps = [i[0] for i in index]
        y = self.__rowiseMeanShapes
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(nbVars, y)
        axs[0].xlabel = "Nombre de variables [-]"
        axs[1].plot(nbTimeSteps, y)
        axs[1].xlabel = "Nombre de pas de temps [-]"
        axes[0].ylabel = f"Moyenne temps (sur {nbRuns} exécutions) [s]"
        if savefig:
            if figname is None:
                figname = "tempsFonctionShape.png"
            fig.savefig(figname)
        if show:
            plt.show()

    def plotTimeFctTauMax(self, show: bool = True, savefig: bool = True, figname: str = None):
        nbRuns = len(self.__dataframeTaus.columns) - 1
        plt.plot(self.__rowiseMeanTaus)
        plt.xlabel(r"$\tau_{max}$ [-]")
        plt.ylabel(f"Moyenne temps (sur {nbRuns} exécutions) [s]")
        if savefig:
            if figname is None:
                figname = "tempsFonctionTau.png"
            plt.savefig(figname)
        if show:
            plt.show()

    @staticmethod
    def computeRowiseMean(data: typing.Union[np.ndarray, pd.DataFrame]):
        return np.mean(data, 1)

    @staticmethod
    def computeRowiseStandardDeviation(data: typing.Union[np.ndarray, pd.DataFrame]):
        return np.std(data, 1)


class BenchmarkStats(BaseBenchmarkStats):

    def __init__(self, benchmark: Benchmark):
        dfs = benchmark.completDataframes
        if dfs is None:
            raise ValueError("No benchmark information.")
        super(BenchmarkStats, self).__init__(*dfs)


class BenchmarkStatsFromFiles(BaseBenchmarkStats):

    def __init__(self, shapesTimeFilename: str, tausTimeFilename: str):
        dfs = (pd.read_csv(shapesTimeFilename, index_col=0), pd.read_csv(tausTimeFilename, index_col=0))
        super(BenchmarkStatsFromFiles, self).__init__(*dfs)


if __name__ == '__main__':
    shapes = [(50, 10), (100, 10), (200, 10), (300, 10), (400, 10), (440, 10), (50, 20), (100, 20), (200, 20),
              (300, 20), (400, 20), (440, 20), (50, 40), (100, 40), (200, 40), (300, 40), (400, 40), (440, 40),
              (50, 50), (100, 50), (200, 50), (300, 50), (400, 50), (440, 50), (50, 100), (100, 100), (200, 100),
              (300, 100), (400, 100), (440, 100), (50, 200), (100, 200), (200, 200),
              (300, 200), (400, 200), (440, 200), (50, 300), (100, 300), (200, 300),
              (300, 300), (400, 300), (440, 300), (50, 300), (100, 300), (200, 300),
              (300, 300), (400, 300), (440, 300)]
    taus = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100, 150]
    datashape = (400, 20)

    path = os.path.join(os.getcwd(), "tigramite", "data", "timeSeries_ax1.npy")
    data = np.load(path).T

    b = Benchmark(data, False, ("shapes.txt", "tauMax.txt"))
    b.start(3, shapes, taus, datashape)

    bstats = BenchmarkStats(b)
    bstats.plotTimeFctShape(False, True)
    bstats.plotTimeFctTauMax(False, True)

    b = Benchmark(data, True, ("shapes.txt", "tauMax.txt"))
    b.start(3, shapes, taus, datashape)

    bstats = BenchmarkStats(b)
    bstats.plotTimeFctShape(False, True, "tempsFonctionTau_par.png")
    bstats.plotTimeFctTauMax(False, True, "ShapeFonctionTau_par.png")
