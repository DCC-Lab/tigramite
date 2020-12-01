import run_pcmci_parallel_v2
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
import os

__nbColsMax__ = 3


class P_ValuesTensorVisualization:

    def __init__(self, p_valuesTensor: np.ndarray, alpha: float = 0.01, tau_min: int = 0):
        self.__pvalues = p_valuesTensor
        if self.__pvalues.ndim != 3:
            raise ValueError("The array must be in 3D.")
        self.__nbSlices = self.__pvalues.shape[-1]
        self.__nbVars = self.__pvalues.shape[0]
        self.__pvaluesRavel = [self.__pvalues[:, :, i].ravel() for i in range(self.__nbSlices)]
        if alpha < 0:
            raise ValueError("The parameter alpha cannot be negative.")
        self.__alpha = alpha
        if tau_min < 0:
            raise ValueError("The parameter tau_min cannot be negative.")
        self.__tau_min = tau_min
        self.__tickLabels = range(self.__nbVars)
        self.__pvalMin, self.__pvalMax = np.min(self.__pvaluesRavel), np.max(self.__pvaluesRavel)

    @classmethod
    def nbCols(cls, nbSlices: int):
        if nbSlices < 1:
            raise ValueError("There must be at least one slice (one tau).")
        return 3 if nbSlices >= 3 else nbSlices

    @classmethod
    def nbRows(cls, nbSlices: int, returnCurrentNbCols: bool = True):
        currentNbCols = cls.nbCols(nbSlices)
        nbRows = (nbSlices - 1) // currentNbCols + 1
        if returnCurrentNbCols:
            return nbRows, currentNbCols
        return nbRows

    def splitSlices(self, withIndex: bool = True):
        for s in range(self.__nbSlices):
            data = self.__pvalues[:, :, s]
            if withIndex:
                yield s, data
            else:
                yield data

    def sideBySideMatrixPValuesHist(self, nbBins: int = None):
        if nbBins is None:
            nbBins = self.__nbVars
        for index, s in self.splitSlices():
            fig, (matrix, hist) = plt.subplots(1, 2)
            fig.suptitle(r"P-Values for $\tau = {}$".format(index + self.__tau_min))
            image = sns.heatmap(s, ax=matrix, cbar=True, xticklabels=self.__tickLabels, yticklabels=self.__tickLabels)
            matrix.set_title("P-Values matrix")
            histAx = sns.histplot(self.__pvaluesRavel[index], bins=nbBins)
            maxCurrentHist = histAx.get_ylim()[-1]
            hist.plot([self.__alpha] * 100, np.linspace(0, maxCurrentHist, 100), ls="--",
                      label=r"$\alpha = {}$".format(self.__alpha))
            hist.legend()
            hist.set_title("P-Values histogram")
            plt.show()

    def allSlicesOnSameFig(self):
        nbRows, nbCols = self.nbRows(self.__nbSlices)
        fig, axes = plt.subplots(nbRows, nbCols)
        fig.suptitle(f"P-Values matrices for {self.__nbVars} variables")
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        index = 0
        for row in range(nbRows):
            for col in range(nbCols):
                currentAxis = axes[row][col]
                sns.heatmap(self.__pvalues[:, :, index], ax=currentAxis, cbar=row == 0, xticklabels=self.__tickLabels,
                            yticklabels=self.__tickLabels, cbar_ax=None if row else cbar_ax, vmin=0, vmax=1)
                currentAxis.set_title(r"$\tau = {}$".format(index + self.__tau_min))
                index += 1
        plt.show()

    def allPValuesHistOnSameFig(self, nbBins: int = None):
        if nbBins is None:
            nbBins = self.__nbVars
        nbRows, nbCols = self.nbRows(self.__nbSlices)
        fig, axes = plt.subplots(nbRows, nbCols)
        fig.suptitle(f"P-Values histograms for {self.__nbVars} variables")
        index = 0
        for row in range(nbRows):
            for col in range(nbCols):
                currentAxis = axes[row][col]
                sns.histplot(self.__pvaluesRavel[index], bins=nbBins, ax=currentAxis)
                currentAxis.set_title(r"$\tau = {}$".format(index + self.__tau_min))
                maxCurrentHist = currentAxis.get_ylim()[-1]
                currentAxis.plot([self.__alpha] * 100, np.linspace(0, maxCurrentHist, 100), ls="--",
                                 label=r"$\alpha = {}$".format(self.__alpha))
                currentAxis.legend()
                index += 1
        plt.show()

    def allPValuesHistOnSamePlot(self, colormap: str = "Spectral", nbBins: int = None):
        cmap = cm.get_cmap(colormap)
        values = np.linspace(0, 1, self.__nbSlices)
        colors = [cmap(value) for value in values]
        if nbBins is None:
            nbBins = self.__nbVars
        fig, ax = plt.subplots()
        for i in range(self.__nbSlices):
            sns.histplot(self.__pvaluesRavel[i], ax=ax, alpha=.25, bins=nbBins,
                         label=r"$\tau = {}$".format(i + self.__tau_min), color=colors[i], binrange=(0, 1))
        maxCurrentHist = ax.get_ylim()[-1]
        ax.plot([self.__alpha] * 100, np.linspace(0, maxCurrentHist, 100), ls="--",
                label=r"$\alpha = {}$".format(self.__alpha))
        ax.legend()
        plt.show()


class P_ValuesMatrixVisualization:

    def __init__(self, p_valuesMatrix: np.ndarray):
        self.__pvalues = p_valuesMatrix
        if self.__pvalues.ndim != 2:
            raise ValueError("The array must be in 2D.")


if __name__ == '__main__':
    # path = os.path.join(os.getcwd(), "tigramite", "data", "timeSeries_ax1.npy")
    # data = np.load(path).T
    # data = data[:440, :10]
    # seq_pcmci = PCMCI(pp.DataFrame(data), ParCorr())
    # results_pcmci_seq = seq_pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.01)
    # p_matrix = results_pcmci_seq["p_matrix"]
    # np.save("test", p_matrix)
    path = r"C:\Users\goubi\Desktop\Maîtrise\AvancementsProjet\500vars440timesteps0tauMin5tauMax.npy"
    p_matrix = np.load(path)
    p = P_ValuesTensorVisualization(p_matrix)
    # p.sideBySideMatrixPValuesHist()
    p.allSlicesOnSameFig()
    p.allPValuesHistOnSameFig()
    #p.allPValuesHistOnSamePlot("gist_ncar")
