import cProfile, pstats
from pstats import SortKey
import numpy as np
from io import StringIO
from tigramite.independence_tests import ParCorr, CMIknn
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
import os


class ProfilerHandler:

    def __init__(self, func: callable, *fargs, **fkwargs):
        self.__func = func
        self.__fargs = fargs
        self.__fkwargs = fkwargs
        self.__profileOutput = StringIO()

    def profile(self, printResult: bool = True):
        profile = cProfile.Profile()
        profile.enable()
        value = self.__func(*self.__fargs, **self.__fkwargs)
        profile.disable()
        stats = pstats.Stats(profile, stream=self.__profileOutput).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        if printResult:
            print(self.__profileOutput.getvalue())
        return value

    def writeStatsToFile(self, filename: str):
        if not filename.endswith(".txt"):
            filename += ".txt"
        with open(filename, "w") as f:
            f.write(self.__profileOutput.getvalue())


class PCMCIProfiler(ProfilerHandler):

    def __init__(self, data: np.ndarray, indepTest, tau_max=5, pc_alpha=0.01):
        dataframe = pp.DataFrame(data)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=indepTest)
        super(PCMCIProfiler, self).__init__(pcmci.run_pcmci, tau_max=tau_max, pc_alpha=pc_alpha)


class PCMCIProfilerLongVersion(PCMCIProfiler):

    def __init__(self):
        path = os.path.join(os.getcwd(), "tigramite", "data", "timeSeries_ax1.npy")
        data = np.load(path).T
        data = data[:440, :30]
        cond_ind_test = ParCorr()
        super(PCMCIProfilerLongVersion, self).__init__(data, cond_ind_test)


if __name__ == '__main__':
    runLongVersion = input("Run the long version (more than 30 minutes)? 'Y' to proceed.\n")
    if runLongVersion == "Y":
        pcmciProfileLong = PCMCIProfilerLongVersion()
        pcmciProfileLong.profile(False)
        pcmciProfileLong.writeStatsToFile("longProfiling.txt")
    else:
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

        data, nonstat = pp.structural_causal_process(links,
                                                     T=1000, seed=7)
        cond_ind_test = ParCorr()
        pcmciProfile = PCMCIProfiler(data, cond_ind_test)
        pcmciProfile.profile()
        pcmciProfile.writeStatsToFile("profileGrapheExemple.txt")
