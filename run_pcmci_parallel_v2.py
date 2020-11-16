"""
Parallel PCMCI: new version using Python's multiprocessing module instead of MPI, which is another dependency.
"""

import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import multiprocessing as mp
import time


class PCMCI_Parallel:
    """
    Variant of the PCMCI class using parallelism for faster computations. This class requires the base PCMCI algorithm
    provided in the tigramite package developed by J. Runge et al. Python's multiprocessing module is also required in
    order to use different cores to perform the PC algorithm as well as the MCI algorithm.

    Parameters
    ----------
    data: NumPy array.
        This is the array containing time series. It must be in the shape (T,N). It is converted to a Tigramite
        DataFrame object in order to be able to use the PCMCI algorithm.
    cond_ind_test: conditional independence test object.
        This can be any class from ``tigramite.independance_tests``.

    Attributes
    ----------
    all_parents: dictionary
        Dictionary of the form {variable:[(parent var, -tau), ...], other variable:[], ...} containing the
        conditioning parents estimated with PC algorithm.
    val_min: dictionary
        Dictionary of the form val_min[j][(i, -tau)] = float containing the minimum test statistic value for each link
        estimated in the PC algorithm
    pval_max: dictionary
        Dictionary of the form pval_max[j][(i, -tau)] = float containing the maximum p-value for each link
        estimated in the PC algorithm
    """

    def __init__(self, data: np.ndarray, cond_ind_test: object, tau_min: int, tau_max: int, pc_alpha: float):
        self.__nbVar = data.shape[-1]
        self.__data = data
        self.__cond_ind_test = cond_ind_test
        self.__tau_max = tau_max
        self.__tau_min = tau_min
        self.__pc_alpha = pc_alpha
        self.all_parents = {}
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test)
        self.__allSelectedLinks = pcmci_var._set_sel_links(None, self.__tau_min, self.__tau_max, False)
        self.__currentSelectedLinks = {key: [] for key in self.__allSelectedLinks.keys()}
        self.pval_max = {}
        self.val_min = {}

    @staticmethod
    def split(container, count):
        """
        Method used to split a container in ``count`` smaller containers.
        Parameters
        ----------
        container: An iterable
            Container to split.
        count: An integer (positive)
            Number of smaller containers to create from the original container.

        Returns
        -------
        splitted_container: A list
            A list of smaller containers. The number of containers in the list is ``count``.
        """
        container = tuple(container)
        return [container[i::count] for i in range(count)]

    def run_pc_stable_parallel_on_selected_variables(self, variables):
        """
        Method used to compute the PC_1 algorithm on selected variables. This method is run by a single process
        in order to use parallelism. For example, if one has 10 variables and 5 cpu cores, each core will run the
        PC_1 algorithm on 2 variables.

        Parameters
        ----------
        variables: An iterable
            An iterable containing a portion of the original variables. These variables will be fed to a single process
            to compute the PC algorithm on each variable.

        Returns
        -------
        out: A list
            A list containing a list for each variable used in this method. Each list has the current variable, its
            PCMCI object and its parents.
        """
        out = []
        pcmci_var = PCMCI(dataframe=pp.DataFrame(self.__data.copy()), cond_ind_test=self.__cond_ind_test)
        for variable in variables:
            parents_of_var = pcmci_var._run_pc_stable_single(variable, tau_min=self.__tau_min, tau_max=self.__tau_max,
                                                             pc_alpha=self.__pc_alpha,
                                                             selected_links=self.__allSelectedLinks)
            out.append([variable, pcmci_var, parents_of_var])
        return out

    def run_mci_parallel_singleVar(self, stuff):
        """
        Parameters
        ----------
        stuff: A list
            A list containg a list for each variable on which the MCI algorithm will be performed. Each list
            must contain the current variable, the current variable's PCMCI object and the current variable's parents.

        Returns
        -------
        out: A list
            Returns a list containing a list for each variable on which the MCI algorithm was performed. Each list
            contain the variable, its PCMCI object, its parents and its specific val_matrix and p_matrix
            (see ``run_mci`` in the ``PCMCI`` class for more info about val_matrix and p_matrix)
        """
        out = []
        for variable, pcmci_var, parents_of_var in stuff:
            currentSelectedLinks = self.__currentSelectedLinks.copy()
            currentSelectedLinks[variable] = self.__allSelectedLinks[variable]
            results_in_var = pcmci_var.run_mci(tau_min=self.__tau_min, tau_max=self.__tau_max, parents=self.all_parents,
                                               selected_links=currentSelectedLinks)
            out.append([variable, pcmci_var, parents_of_var, results_in_var])
        return out

    def start(self, nbWorkers: int = None):
        if nbWorkers is None:
            nbWorkers = mp.cpu_count()
        if nbWorkers > mp.cpu_count():
            nbWorkers = mp.cpu_count()

        if nbWorkers > self.__nbVar:
            nbWorkers = self.__nbVar
        splittedJobs = self.split(range(self.__nbVar), nbWorkers)
        with mp.Pool(nbWorkers) as pool:
            pc_output = pool.map(self.run_pc_stable_parallel_on_selected_variables, splittedJobs)

        for elem in pc_output:
            for innerElem in elem:
                self.all_parents.update(innerElem[-1])
        pc_output = self.split(pc_output, nbWorkers)

        with mp.Pool(nbWorkers) as pool:
            output = pool.starmap(self.run_mci_parallel_singleVar, pc_output)

        return output


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

    pcmci = PCMCI(pp.DataFrame(data), ParCorr())
    start = time.time()
    results = pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.01)
    # print(pcmci.all_parents)
    print(f"Total time: {time.time() - start}")
    pcmci_par = PCMCI_Parallel(data, ParCorr(), 0, 5, 0.01)
    start = time.time()
    pcmci_par.start()
    # print(pcmci_par.all_parents)
    print(f"Total time: {time.time() - start}")
    print(sorted(pcmci_par.all_parents) == sorted(pcmci.all_parents))

    print(pcmci.all_parents)
    p_val = results["p_matrix"]
    import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(2, 3)
    # currSlice = 0
    # for row in range(2):
    #     for col in range(3):
    #         axes[row, col % 3].imshow(p_val[:, :, currSlice])
    #         axes[row, col % 3].set_title(rf"$\tau$ = {-currSlice}")
    #         currSlice += 1
    # plt.show()
