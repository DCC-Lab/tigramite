import tigramtite as tg
import run_pcmci_parallel
import time
import os
import numpy as np
import matplotlib.pyplot
from tigramite.independence_tests import ParCorr, CMIknn
import tigramite.data_processing as pp
import tigramite.plotting as tp
import pandas as pd


class Benchmark:

    def __init__(self, data: np.ndarray):
        self.__data = data
        self.__times = []

    def shapeImpact(self, shapes: list, tau_max: int = 3):
        self.__times.clear()
        total = len(shapes)
        shapesDone = 1
        correctShape = []
        for shape in shapes:
            subdata = self.__data[:shape[0], :shape[1]]
            correctShape.append(subdata.shape)
            start = time.perf_counter_ns()
            self.runPCMCI(subdata, tau_max)
            end = time.perf_counter_ns()
            self.__times.append((start - end) / 1e9)
            self.saveTimes("shapes.txt", correctShape)
            shapesDone += 1
            print(f"Shapes done: {shapesDone} / {total}")

    def tau_maxImpact(self, tau_max_s: list, dataShape: tuple):
        self.__times.clear()
        subdata = self.__data[:dataShape[0], :dataShape[1]]
        total = len(tau_max_s)
        tauDone = 1
        tausDoneList = []
        for tau in tau_max_s:
            start = time.perf_counter_ns()
            self.runPCMCI(subdata, tau)
            end = time.perf_counter_ns()
            self.__times.append((start - end) / 1e9)
            tausDoneList.append(tau)
            self.saveTimes("tauMax.txt", tausDoneList)
            tauDone += 1
            print(f"Tau done: {tauDone} / {total}")

    def runPCMCI(self, data: np.ndarray, tau_max: int):
        dataframe = pp.DataFrame(data)
        cond_ind_test = ParCorr()
        pcmci = tg.pcmci.PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=0.01)

    def saveTimes(self, fname: str, columns: list):
        if not fname.endswith(".txt"):
            fname += ".txt"
        df = pd.DataFrame(self.__times, columns=columns)
        df.to_csv(fname)


if __name__ == '__main__':
    shapes = [(10, 10), (100, 10), (None, 10), (10, 20), (100, 20), (None, 20)]
    taus = [2, 3, 4, 5, 8, 10]
    datashape = (400, 10)

    path = os.path.join(os.getcwd(), "data", "timeSeries_ax1.npy")
    data = np.load(path).T

    b = Benchmark(data)
    b.shapeImpact(shapes)
    b.tau_maxImpact(taus, datashape)
