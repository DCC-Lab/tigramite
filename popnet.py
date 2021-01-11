"""PopNet helps to study the Wilson--Cowan model and some of its extensions.

The PopNet module is made to perform various numerical experiments related to
the study of the Wilson--Cowan model and its extensions that include explicitely
a refractory state and covariances between different fractions of populations.
It can both perform numerical integrations of dynamical systems related to
Wilson--Cowan's, or perform simulations of the stochastic process which rules
the underlying dynamics of a network whose behavior can macroscopically be
represented by the Wilson--Cowan model.

PopNet provides methods to set, save or load all parameters used for an
experiment, others to actually perform the experiment. It also has a number of
methods, based on [`matplotlib`](https://matplotlib.org/), to display the
outputs of experiments. The [Classes And Hierarchy](#classes-and-hierarchy)
section below gives a summary of the important classes of the module. A complete
example of a numerical integration is given in the [Example](#example) section
below.

Classes and hierarchy
---------------------
The important classes of the module are summarized below. The indentation
follows the hierarchy.

 - `Population` : Represent a population of biological neurons.
 - `Network` : Represent a biological neural network split into populations. 
     - `MicroNetwork` : Represent a network including its microscopic structure.
 - `Configuration` : A complete configuration to perform a numerical experiment.
     - `ConfigurationOne` : A configuration with a network of one population.
     - `MicroConfiguration` : A configuration with a `MicroNetwork`.
     - `MicroConfigurationOne` : A merge of the two above classes.
 - `Executor` : An interface to run a numerical experiment.
     - `Integrator` : An interface to run numerical integrations. 
         - `SimpleIntegrator` : An integrator for the Wilson--Cowan dynamics.
         - `ExtendedIntegrator` : An integrator for the *extended* Wilson--Cowan
           dynamics.
         - `ExtendedIntegratorOne` : Special case of the last for a network of
           one population.
     - `Simulator` : An interface to run stochastic process simulations.
         - `SimpleSimulator` : A simulator to run simulations one at a time.
         - `ChainSimulator` : A simulator to run multiple simulations at once.
 - `Result` : Represent results of numerical experiments.
     - `MeanField` : Output given by `SimpleIntegrator`.
     - `Solution` : Output given by `ExtendedIntegrator`.
     - `Trajectory` : Output given by `Simulator`.
     - `Statistics` : Output given by `ChainSimulator`.
     - `Spectrum` : Fourier transform of another result.

`Result` and all of its subclasses also have a `One` suffixed version dedicated
to handle the case where the network used has a single population.

Conventions
-----------
In all of the PopNet module, every `Population`, `Network` of `Configuration`
instance is given an identificator (ID). These IDs are meant to ease the task
of organizing data. To ensure that there is no confusion among saved files, some
conventions are to be followed:

 - A `Population`'s ID is always a single character;
 - A `Network`'s ID always begins with its number of populations;
 - A `Configuration`'s ID always begins with that of its network.

Errors are raised when they are not followed.

Example
-------
In the following example, we perform a numerical integration of the dynamical
system from the extended Wilson--Cowan model. First, we import PopNet and we
define a population, which is used to define a network. The parameters of the
network are saved.

>>> import popnet as pn
>>> 
>>> pop = pn.Population('Population')
>>> pop.alpha = .07
>>> pop.beta = 1.
>>> pop.gamma = .5
>>> net = pn.Network('1A', pop)
>>> net.c = -2.6
>>> net.save()

Then, a configuration is defined from the network created before. Again, the
configuration is saved.

>>> config = pn.make_config(net)
>>> config.initial_state = [.5, .3, .16, .07, -.085]
>>> config.save()

The configuration is used to perform a numerical integration of the dynamical
system, from the given initial state.

>>> integrator = pn.get_integrator(config, system='extended')
>>> integrator.run()

The output of the integration is then shown graphically, and saved.

>>> solution = integrator.output()
>>> solution.initialize_graph()
>>> solution.plot_expectations()
>>> solution.plot_variances()
>>> solution.setup_graph(time_units='$1/\\beta$')
>>> solution.make_legend(ncol=2)
>>> solution.end_graph(savefig=True)

"""

import ast
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import ode
from tqdm import tqdm
from copy import deepcopy
from warnings import warn


class PopNetError(Exception):
    """Generic class for exceptions relative to the PopNet module."""
    pass


class FormatError(PopNetError):
    """Exceptions relative to formatting errors in files handled by PopNet."""
    pass


class PopNetWarning(Warning):
    """Generic class for warnings relative to the PopNet module."""
    pass


class PopNetDict(dict):
    """Modified dictionary used by PopNet classes.

    A `PopNetDict` is a dictionary which expects its keys to be strings and its
    values to be lists, or lists of lists forming a square array if the key
    begins with "C". The idea is to assign values to state variables of
    populations in a network. A `PopNetDict` will not accept non-string keys,
    and will place all values in a list of the correct form if possible. If it
    is not an error will be raised. 

    Examples
    --------
    In the following example, a `PopNetDict` is first created with two valid
    keys.

    >>> D = PopNetDict({'A': 1, 'CAA': 2})
    >>> D
    {'A': [1], 'CAA': [[2]]}

    If new non-list values are assigned, they will still be placed in the
    expected list format. 

    >>> D['R'] = 3
    >>> D['CRR'] = 4
    >>> D
    {'A': [1], 'CAA': [[2]], 'R': [3], 'CRR': [[4]]}

    It is not possible to add a key to the dictionary if it is not a string.
    Trying to do so will raise an error.

    >>> D[1] = 5
    PopNetError: The keys of a PopNetDict should be strings.

    It is neither possible to set as a value a list of lists if it is not
    formatted as a square array, that is, if the lists contained in the list do
    not have the same length as the list itself. 

    >>> D['CSS'] = [1, 2]
    PopNetError: The values of a PopNetDict should be lists or lists of 
    lists forming a square array.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in self:
            self._key_check(key)
            self[key] = self._value_check(key, self[key])

    def __setitem__(self, key, value):
        self._key_check(key)
        value = self._value_check(key, value)
        super().__setitem__(key, value)

    @staticmethod
    def _key_check(key):
        if not isinstance(key, str):
            raise PopNetError('The keys of a PopNetDict should be strings.')

    @staticmethod
    def _value_check(key, value):
        if key[0] == 'C':
            if not isinstance(value, list):
                value = [[value]]
            else:
                for j, list_value in enumerate(value):
                    if not isinstance(list_value, list):
                        value[j] = [list_value]
            for list_value in value:
                if len(list_value) != len(value):
                    raise PopNetError('The values of a PopNetDict should be '
                                      'lists or lists of lists forming a square'
                                      ' array.')
        else:
            if not isinstance(value, list):
                value = [value]
        return value


class Population:
    """Represent a population of biological neurons.

    This class is used to describe a population of biological neurons. It allows
    to easily attribute parameters, such as the threshold and the transition
    rates, to the same population. Its methods allow to change easily the values
    of these parameters. 

    Parameters
    ----------
    name : str
        Name of the population.
    ID : str, optional
        Identificator of the population. It must be a single character. If it is
        not given, the last character of `name` is used if it is a number, and
        else the first one is used.

    Attributes
    ----------
    name : str
        Name given to the population. See `Population.name`.
    ID : str
        Identificator of the population. See `Population.ID`.
    size : int
        Size of the population. See `Population.size`.
    alpha : float
        Mean transition rate from sensitive to active. See `Population.alpha`.
    beta : float
        Mean transition rate from active to refractory. See `Population.beta`.
    gamma : float
        Mean transition rate from refractory to sensitive. See
        `Population.gamma`.
    theta : float
        Mean threshold. See `Population.theta`.
    scale_alpha : float
        Scale of transition rates from sensitive to active. See
        `Population.scale_alpha`.
    scale_beta : float
        Scale of transition rates from active to refractory. See
        `Population.scale_beta`.
    scale_gamma : float
        Scale of transition rates from refractory to sensitive. See
        `Population.scale_gamma`.
    scale_theta : float
        Scale of thresholds. See `Population.scale_theta`.

    """

    def __init__(self, name, ID=None):
        self.name = name
        if ID is None:
            ID = self._default_ID()
        self.ID = ID

        self.size = None
        self._means = {'alpha': 1., 'beta': 1., 'gamma': 1., 'theta': 0.}
        self._scales = {'alpha': 0., 'beta': 0., 'gamma': 0., 'theta': 1.}
        self._update_means()
        self._update_scales()

    def __str__(self):
        string = f'{self.name}'
        if self.size is not None:
            string += f' - {self.size} neurons'
        for mean in self._means:
            string += f'\n{mean:>11} = {self._means[mean]}'
        for scale in self._scales:
            scale_name = 'scale ' + str(scale)
            string += f'\n{scale_name:>11} = {self._scales[scale]}'
        return string

    @property
    def name(self):
        """Name of the population. 

        Name given to the population to identify it in a network. It has to be
        a string. When setting a new name, if the population's ID was the
        default one, then the ID is updated according to the new name.
        """
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError('A population\'s name must be a string.')
        try:
            assert self.ID == self._default_ID()
        except (AttributeError, AssertionError):
            pass
        else:
            self.ID = self._default_ID(new_name)
        self._name = new_name

    @property
    def ID(self):
        """ID of the population. 

        Identificator given to the population. It must be a single character,
        else an error is raised when setting it. The ID is notably used as a
        subscript to identify state variables. 
        """
        return self._ID

    @ID.setter
    def ID(self, new_ID):
        if len(str(new_ID)) != 1:
            raise PopNetError('A population\'s ID must be a single character.')
        self._ID = str(new_ID)

    @property
    def size(self):
        """Number of neurons in the population. 

        Number of neurons in the population. It must be a positive integer.
        """
        return self._size

    @size.setter
    def size(self, new_value):
        if new_value is None:
            self._size = None
            return
        try:
            new_value = int(new_value)
        except TypeError as error:
            raise TypeError('A population\'s size must be a number.') from error
        else:
            if not new_value > 0:
                raise ValueError('A population\'s size must be positive.')
        self._size = new_value

    @property
    def alpha(self):
        """Mean `alpha` transition rate in the population.

        Mean value of the transition rates from sensitive to active (with
        sufficient input) in the population. 
        """
        return self._alpha

    @alpha.setter
    def alpha(self, new_value):
        self._means['alpha'] = float(new_value)
        self._alpha = float(new_value)

    @property
    def beta(self):
        """Mean `beta` transition rate in the population.

        Mean value of the transition rates from active to refractory in the
        population.
        """
        return self._beta

    @beta.setter
    def beta(self, new_value):
        self._means['beta'] = float(new_value)
        self._beta = float(new_value)

    @property
    def gamma(self):
        """Mean `gamma` transition rate in the population.

        Mean value of the transition rates from refractory to sensitive in the
        population. 
        """
        return self._gamma

    @gamma.setter
    def gamma(self, new_value):
        self._means['gamma'] = float(new_value)
        self._gamma = float(new_value)

    @property
    def theta(self):
        """Mean threshold in the population. 

        Mean value of the thresholds of activation in the population. 
        """
        return self._theta

    @theta.setter
    def theta(self, new_value):
        self._means['theta'] = float(new_value)
        self._theta = float(new_value)

    @property
    def scale_alpha(self):
        """Scaling factor of the `alpha` transition rate in the population.

        Scaling factor of the distribution of the transition rates from
        sensitive to active (with sufficient input) in the population, which is
        assumed to be a logistic distribution.
        """
        return self._scale_alpha

    @scale_alpha.setter
    def scale_alpha(self, new_value):
        self._scales['alpha'] = float(new_value)
        self._scale_alpha = float(new_value)

    @property
    def scale_beta(self):
        """Scaling factor of the `beta` transition rate in the population.

        Scaling factor of the distribution of the transition rates from
        active to refractory in the population, which is assumed to be a
        logistic distribution.
        """
        return self._scale_beta

    @scale_beta.setter
    def scale_beta(self, new_value):
        self._scales['beta'] = float(new_value)
        self._scale_beta = float(new_value)

    @property
    def scale_gamma(self):
        """Scaling factor of the `gamma` transition rate in the population.

        Scaling factor of the distribution of the transition rates from
        refractory to sensitive in the population, which is assumed to be a
        logistic distribution. 
        """
        return self._scale_gamma

    @scale_gamma.setter
    def scale_gamma(self, new_value):
        self._scales['gamma'] = float(new_value)
        self._scale_gamma = float(new_value)

    @property
    def scale_theta(self):
        """Scaling factor of the thresholds in the population.

        Scaling factor of the distribution of the thresholds in the population,
        which is assumed to be a logistic distribution.
        """
        return self._scale_theta

    @scale_theta.setter
    def scale_theta(self, new_value):
        self._scales['theta'] = float(new_value)
        self._scale_theta = float(new_value)

    def copy(self, name, ID=None):
        """Copy the population.

        Return a copy of the population with a new name and ID.

        Parameters
        ----------
        name : str
            Name to give to the new population.
        ID : str, optional
            ID to give to the new population. Defaults to `None`, in which case a
            default one is taken from the name.

        Returns
        -------
        Population
            The copied population.
        """
        other = deepcopy(self)
        other.name = name
        if ID is None:
            ID = other._default_ID()
        other.ID = ID
        return other

    def F(self, y):
        """Cumulative distribution function of the thresholds.

        Cumulative distribution function (CDF) of the thresholds in the
        population, assuming they follow a logistic distribution of mean `theta`
        and of scale factor `scale_theta`. 
        """
        return 1 / (1 + np.exp( - (y - self.theta) / self.scale_theta))

    def dF(self, y):
        """First derivative of the thresholds' CDF. See `Population.F`."""
        return 1 / self.scale_theta * self.F(y) * (1 - self.F(y))

    def ddF(self, y):
        """Second derivative of the thresholds' CDF. See `Population.F`."""
        return 1 / self.scale_theta**2 * ( 
                                self.F(y) * (1 - self.F(y)) * (1 - 2*self.F(y)))

    def dddF(self, y):
        """Third derivative of the thresholds' CDF. See `Population.F`."""
        return 1 / self.scale_theta**3 * (
               self.F(y) * (1 - self.F(y)) * (1 - 6*self.F(y) + 6*self.F(y)**2))

    def set_means(self, **new_values):
        """Set the means of population parameters.

        Assign new values to some of the means of the population's parameters:
        `alpha`, `beta`, `gamma` and `theta`.

        Parameters
        ----------
        **new_values
            New values to assign to means of valid population parameters. 

        Raises
        ------
        KeyError
            If non-valid parameters are requested.
        """
        for key in new_values:
            if key not in self._means:
                raise KeyError(f'{key} is not a valid population parameter.')
            self._means[key] = float(new_values[key])
        self._update_means()

    def set_random_rates(self, rates=None, distribution='exponential', **kwargs):
        """Randomly set the transition rates from a given distribution.

        Choose a random value for the transition rates from a given distribution
        family with given parameters. The random value is generated using a
        [`Generator`](https://tinyurl.com/numpy-random-generator) instance from
        NumPy's `random` module, and keyword arguments can be passed to the
        `Generator`'s method used to generate the random values.

        Parameters
        ----------
        rates : list or tuple of str, or str, optional
            The transition rates that should be chosen randomly. It should
            contain valid rates in `alpha`, `beta` or `gamma`, or be one of
            these strings. 
        distribution : {'uniform', 'exponential'}, optional
            The distribution family used to choose a value for the rates.
            Defaults to `'uniform'`. 
        **kwargs
            Keyword arguments to be passed to the method of `Generator`
            corresponding to the correct distribution family. 

        Raises
        ------
        TypeError
            If `rates` is not a list, tuple, or str.
        KeyError
            If the strings given in `rates` are not valid transition rates.
        NotImplementedError
            If the requested distribution family is not implemented.
        """
        if rates is None:
            rates = ['alpha', 'beta', 'gamma']
        if isinstance(rates, str):
            rates = [rates]
        if not isinstance(rates, (list, tuple)):
            raise TypeError('Population.set_random_rates expects its first '
                            'argument to be either a list of rates given as '
                            'strings, or a single rate given as a string.')
        for rate in rates:
            if rate not in ['alpha', 'beta', 'gamma']:
                raise KeyError(f'{rate} is not a valid transition rate.')
            rng = np.random.default_rng()
            if distribution == 'uniform':
                self._means[rate] = rng.uniform(**kwargs)
                continue
            elif distribution == 'exponential':
                self._means[rate] = rng.exponential(**kwargs)
                continue
            raise NotImplementedError(f'No {distribution} distribution available'
                                      ' to randomly set a transition rate.')
        self._update_means()

    def set_random_threshold(self, distribution='uniform', **kwargs):
        """Randomly set the threshold from a given distribution.

        Choose a random value for the threshold from a given distribution family
        with given parameters. The random value is generated using a
        [`Generator`](https://tinyurl.com/numpy-random-generator) instance from
        NumPy's `random` module, and keyword arguments can be passed to the
        `Generator`'s method used to generate the random values.

        Parameters
        ----------
        distribution : {'uniform'}, optional
            The distribution family used to choose a value for the threshold.
            Defaults to `'uniform'`, which is for now the only implemented
            distribution. 
        **kwargs
            Keyword arguments to be passed to the method of `Generator`
            corresponding to the correct distribution family. 

        Raises
        ------
        NotImplementedError
            If the requested distribution family is not implemented. 
        """
        if distribution == 'uniform':
            self.theta = np.random.default_rng().uniform(**kwargs)
            return
        raise NotImplementedError(f'No {distribution} distribution available '
                                  'to randomly set a threshold.')

    def set_scales(self, **new_values):
        """Set the scales of population parameters.

        Assign new values to some of the scales of the population's parameters:
        `alpha`, `beta`, `gamma` and `theta`.

        Parameters
        ----------
        **new_values
            New values to assign to scales of valid population parameters. 

        Raises
        ------
        KeyError
            If non-valid parameters are requested.
        """
        for key in new_values:
            if key not in self._scales:
                raise KeyError(f'{key} is not a valid population parameter.')
            self._scales[key] = float(new_values[key])
        self._update_scales()

    def _default_ID(self, name=None):
        """Get a default ID based on `name`.

        Get a default ID based on the name `name`. If `name` ends with a number,
        this number is returned. Else, the first letter of `name` is returned.

        Parameters
        ----------
        name : str, optional
            Name from which to get an ID. Defaults to `None`, in which case the
            `name` attribute is used instead.

        Returns
        -------
        str
            The said default ID.
        """
        if name is None:
            name = self.name
        assert isinstance(name, str), '\'name\' argument should be a string.'
        try:
            no = int(name[-1])
        except:
            return name[0]
        else:
            return str(no)

    @classmethod
    def _load(cls, lines):
        """Load a population.

        Load a population's parameters from a list of strings, which should be
        the lines of a string representation of a `Population` instance.

        Parameters
        ----------
        lines : list of str
            Strings from which the parameters are to be set. It should be the
            lines of a string representation of a `Population` instance.

        Returns
        -------
        Population
            The loaded population.

        Raises
        ------
        KeyError
            If `string` contains assignments to non-valid parameters.
        FormatError
            If `string` does not have the expected format. 
        """
        lines = [line.replace('\n', '') for line in lines]
        name_line = lines[0].split('-')
        pop = Population(name_line[0].strip())
        if len(name_line) == 2:
            pop.size = name_line[-1].strip().split()[0]
        for line in lines[1:]:
            # Take the part of the line specifying the parameter to set.
            param_spec = line[:11].strip().split()
            if (n := len(param_spec)) == 0:
                # If there is nothing on the line, continue to the next one.
                continue
            if n == 1:
                # If there is a single word, it should give a parameter's mean.
                param = param_spec[0]
                if param not in pop._means:
                    raise KeyError(f'{param} is not a valid parameter.')
                pop._means[param] = float(line[14:])
            elif n == 2:
                # If there is two words, it should give a parameter's scale.
                scale = param_spec[0]
                param = param_spec[1]
                if scale != 'scale' or param not in pop._scales:
                    raise KeyError(f'{scale} {param} is not a valid parameter.')
                pop._scales[param] = float(line[14:])
            else:
                # If there more than two words, something is wrong...
                raise FormatError('It seems that the string cannot be used to '
                                  'define the parameters of a Population '
                                  'instance.')
        pop._update_means()
        pop._update_scales()
        return pop

    def _update_means(self):
        """Update the parameters' means according to `_means` values.

        Update the attributes `alpha`, `beta`, `gamma` and `theta` according to
        the values of the corresponding entries of `._means`. It is intended to
        be used internally in other methods when setting new values to
        parameters, to ensure that the values in `_means` are consistent
        with the values of the corresponding attributes.
        """
        self.alpha = self._means['alpha']
        self.beta  = self._means['beta']
        self.gamma = self._means['gamma']
        self.theta = self._means['theta']

    def _update_scales(self):
        """Update the parameters' scales according to `_scales` values.

        Update the attributes `scale_alpha`, `scale_beta`, `scale_gamma` and
        `scale_theta` according to the values of the corresponding entries of
        `_scales`. It is intended to be used internally in other methods when
        setting new values to parameters, to ensure that the values in `_scales`
        are consistent with the values of the corresponding attributes.
        """
        self.scale_alpha = self._scales['alpha']
        self.scale_beta  = self._scales['beta']
        self.scale_gamma = self._scales['gamma']
        self.scale_theta = self._scales['theta']


class Network:
    """Represent a biological neural network from a macroscopic point of view.

    Represents a biological neural network split into different populations.
    Each population of such a network is expected to be a `Population` instance.
    The purpose of this class is mostly to have a consistent interface to
    define, modify, save, or load the parameters of a network.

    Parameters
    ----------
    ID : str
        ID of the network.
    populations : tuple of Population, or Population
        Defines the populations that constitute the network. Can be given as a
        `Population` instance to make a network with a single population. 

    Attributes
    ----------
    ID : str
        ID of the network. See `Network.ID`.
    populations : tuple of Population
        Populations of the network. See `Network.populations`.
    c : array_like
        Connection matrix. See `Network.c`.
    scale_c : array_like
        Scale of connection weights. See `Network.scale_c`.

    Raises
    ------
    TypeError
        If `populations` cannot be converted to a tuple of `Population`
        instances.

    """

    def __init__(self, ID, populations):
        try:
            self._populations = tuple(populations)
        except TypeError:
            self._populations = (populations,)
        if not all(isinstance(pop, Population) for pop in self._populations):
            raise TypeError('The "populations" attribute of a Network instance '
                            'should be a tuple of Population instances.')
        self.ID = ID
        self.c = np.ones((p := len(self.populations), p))
        self.scale_c = np.zeros((p, p))

    def __str__(self):
        string = f'Network {self.ID}\n\n'
        for pop in self.populations:
            string += str(pop)
            string += '\n\n'
        string += f'Connection matrix:\n{self.c}'
        return string

    @staticmethod
    def load(load_ID, new_ID=None, folder=None):
        """Alias for `load_network`."""
        return load_network(load_ID, new_ID=new_ID, folder=folder)

    @property
    def ID(self):
        """ID of the network.

        Identificator given to the network. Its first character has to be the
        number of populations of the network, else an error is raised when
        setting it. The ID is notably used to name files when saving the network
        parameters.
        """
        return self._ID

    @ID.setter
    def ID(self, new_ID):
        if not isinstance(new_ID, str):
            raise TypeError('The network\'s ID should be a string.')
        if int(new_ID[0]) != len(self.populations):
            raise PopNetError('The first character of the network\'s ID should '
                              'be its number of populations.')
        self._ID = new_ID

    @property
    def populations(self):
        """Populations of the network.

        Tuple containing the populations of the network, given as `Population`
        instances. It is set at initialization, but it cannot be reset nor
        deleted afterwards.
        """
        return self._populations

    @property
    def c(self):
        """Connection matrix of the network.

        Describes the weights of connections between populations of the network.
        The exact relation to the weights of links between individual neurons of
        the network is described in the [Notes](#network-c-notes) section below.
        It has to be a square matrix, but it can be given as a float if the
        network has only one population.

        Notes {#network-c-notes}
        -----
        For clarity, let *J* and *K* be the *j*th and *k*th populations of the
        network respectively, following the order given in the `populations`
        attribute. Then, the element `c[j,k]` of `c` describes the link *from K
        to J*. From the microscopic point of view, it is the product of the size
        of *K* with the mean value of the weights of links from neurons of *K*
        to neurons of *J*. 
        """
        return self._c

    @c.setter
    def c(self, new_c):
        try:
            float_new_c = float(new_c)
        except:
            pass
        else:
            new_c = np.array([[float_new_c]])
        if np.shape(new_c) != (p := len(self.populations), p):
            raise PopNetError('The connection matrix c should be a square '
                              'array whose size corresponds to the number of '
                              'populations of the network.')
        self._c = np.array(new_c, float)

    @property
    def scale_c(self):
        """Scaling factor of the weights' distributions.

        Scaling factors used to define the weights' distributions, which are all
        assumed to be logistic. The exact relation to the weights of links
        between individual neurons of the network is described in the
        [Notes](#network-scale-c-notes) section below. It has to be a square
        matrix, but it can be given as a float if the network has only one
        population.

        Notes {#network-scale-c-notes}
        -----
        If *J* and *K* are respectively the *j*th and the *k*th populations of
        the network, following the order given in the `populations` attribute,
        the actual scaling factor of the *non-zero* weights of links from
        neurons of *K* to neurons of *J* is

        \\[ \\frac{ s_{JK} P_{JK} }{ |K| } \\]

        where \\(s_{JK}\\) is `scale_c[j,k]`, \\(P_{JK}\\) is the probability of
        connection from neurons of *J* to neurons of *K*, and \\(|K|\\) is the
        size of *K*. 
        """
        return self._scale_c

    @scale_c.setter
    def scale_c(self, new_scale):
        try:
            float_new_scale = float(new_scale)
        except:
            pass
        else:
            new_scale = np.array([[float_new_scale]])
        if np.shape(new_scale) != (p := len(self.populations), p):
            raise PopNetError('The scales of the weights should be a square '
                              'array whose size corresponds to the number of '
                              'populations of the network.')
        self._scale_c = np.array(new_scale, float)

    def copy(self, new_ID):
        """Copy the network.

        Return a copy of the network with a new ID. 

        Parameters
        ----------
        new_ID : str
            ID to give to the new network.

        Returns
        -------
        Network
            The copied network.
        """
        other = deepcopy(self)
        other.ID = new_ID
        return other

    def underlying(self):
        """Get the microscopic network underlying the present macroscopic one.

        Return the microscopic network underlying the present macroscopic
        network of populations. The returned network has the same ID, the same
        populations and the same parameters as the present one.

        Returns
        -------
        MicroNetwork
            The underlying microscopic network.
        """
        microself = MicroNetwork(self.ID, self.populations)
        microself.c = self.c
        microself.scale_c = self.scale_c
        microself.reset_parameters()
        return microself

    def save(self, folder=None, note=None):
        """Save the network's parameters in a text file.

        Save the string representation of the network in a text file, under the
        name *ID - Network parameters.txt*, where *ID* is the `ID` attribute.

        Parameters
        ----------
        folder : str, optional
            A folder in which the file will be saved. It should already exist in
            the current directory. Defaults to `None`, in which case the file is
            saved in the current directory.
        note : str, optional
            If given, an additional section "Additional notes:" will be written
            in the file, and `note` will be written there. 
        """
        filename = _format_filename(folder, self.ID, 'Network parameters')
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(self))
            if note is not None:
                file.write('\n\nAdditional notes:\n')
                file.write(note)

    def set_random_c(self, distribution='uniform', signs=None, **kwargs):
        """Randomly set the connection matrix. 

        Choose random values for entries of the connection matrix from a given
        distribution family with given parameters. The random value is generated
        using a [`Generator`](https://tinyurl.com/numpy-random-generator)
        instance from NumPy's `random` module, and keyword arguments can be
        passed to the `Generator`'s method used to generate the random values.

        Parameters
        ----------
        distribution : {'uniform', 'exponential'}, optional
            The distribution family used to choose a value for the threshold. If
            a positive distribution is chosen, the signs of the components of
            `c` are supposed to be fixed by `signs`. Defaults to `'uniform'`. 
        signs : array_like, optional
            A matrix that multiplies the random results. It is intended to be
            used to assign specific signs to the components of `c`. It should be
            a square matrix of -1's and 1's of the same shape as `c`. Defaults
            to `None`, in which case it is replaced by an array of ones. 
        **kwargs
            Keyword arguments to be passed to the method of `Generator` 
            corresponding to the correct distribution family. 

        Raises
        ------
        NotImplementedError
            If the requested distribution family is not implemented. 
        """
        shape = (p := len(self.populations), p)
        if signs is None:
            signs = np.ones(shape)
        else:
            signs = np.array(signs, float)
        rng = np.random.default_rng()
        if distribution == 'uniform':
            self.c = signs * rng.uniform(size=shape, **kwargs)
            return
        elif distribution == 'exponential':
            self.c = signs * rng.exponential(size=shape, **kwargs)
            return
        raise NotImplementedError(f'No {distribution} distribution available '
                                  'to randomly set a connection matrix.')

    def _set_c_from_string(self, string):
        """Set the connection matrix from a string.

        Set the connection matrix `c` from a string.

        Parameters
        ----------
        string : str
            String from which the connection matrix is set. It should have the
            format of a string representation of a NumPy array. 

        Raises
        ------
        FormatError
            If the string does not have the correct format. 
        """
        if string[-1] == '\n':
            string = string[:-1]
        string = re.sub(r'\[\s+', '[', string)
        string = re.sub(r'\s+\]', ']', string)
        string = re.sub(r'\s+', ',', string)
        try:
            new_c = ast.literal_eval(string)
        except:
            raise FormatError('It seems that the string cannot be converted '
                              'to a connection matrix.')
        self.c = new_c


class MicroNetwork(Network):
    """Represent a biological neural network from a microscopic point of view.

    The `MicroNetwork` class extends the `Network` class to characterize
    individual neurons rather than characterizing only their mean values and
    scales by populations. It introduces new attributes to get the values of
    transition rates, thresholds and weights of connection for all neurons.

    The initialization of a `MicroNetwork` is the same as in the base class,
    except that the parameters of individual neurons of the network are also
    initialized. Hence, the size of every population of the network must be
    defined.

    !!! note
        It is important to understand that, even if parameters `alpha`, `beta`,
        `gamma`, `theta` and `W` are generated automatically from the
        corresponding mean values and scaling factors at initialization, it does
        *not* mean that they will be updated upon update of the mean values or
        scaling factors, or upon change in the size of the network. In order to
        remain consistent when new values are set, parameters should be reset
        with `MicroNetwork.reset_parameters`.

    Raises
    ------
    PopNetError
        If the size of a population is not defined.

    """

    def __init__(self, ID, populations):
        super().__init__(ID, populations)
        if any(pop.size is None for pop in populations):
            raise PopNetError('Cannot define a MicroNetwork if the sizes of '
                              'its populations are not defined.')
        self.reset_parameters()

    @property
    def alpha(self):
        """Transition rates from sensitive to active.
        
        Array of transition rates from sensitive to active (with sufficient
        input) of all neurons of the network. It cannot be set nor deleted, but
        it can be reset with `MicroNetwork.reset_parameters`.
        """
        return self._alpha

    @property
    def beta(self):
        """Transition rates from active to refractory.
        
        Array of transition rates from active to refractory of all neurons of
        the network. It cannot be set nor deleted, but can be reset with
        `MicroNetwork.reset_parameters`.
        """
        return self._beta

    @property
    def gamma(self):
        """Transition rates from refractory to sensitive.
        
        Array of transition rates from refractory to sensitive of all neurons of
        the network. It cannot be set nor deleted, but can be reset with
        `MicroNetwork.reset_parameters`.
        """
        return self._gamma

    @property
    def theta(self):
        """Thresholds.
        
        Array of thresholds of all neurons of the network. It cannot be set nor
        deleted, but it can be reset with `MicroNetwork.reset_parameters`.
        """
        return self._theta

    @property
    def W(self):
        """Weight matrix.
        
        Array of weights of connection between neurons of the network. An
        element `W[j,k]` of `W` is the weight of the connection *from* `k` *to*
        `j`. It has to be a real matrix of shape \\(N \\times N\\), where
        \\(N\\) is the size of the network. It cannot be deleted.
        """
        return self._W

    @W.setter
    def W(self, new_value):
        try:
            new_value = np.array(new_value, float)
        except (TypeError, ValueError) as err:
            raise ValueError('A weight matrix must have real entries.') from err
        if new_value.shape != (self.size(), self.size()):
            raise PopNetError('A weight matrix should be square with shape N x '
                              'N, where N is the size of the network.')
        self._W = new_value

    def reset_parameters(self, params=None):
        """Randomly generate the parameters of the network's neurons.

        Generate the parameters that characterize the neurons of the network.
        All parameters are taken from logistic distributions with means and
        scaling factors consistent with the values given by the populations.

        Parameters
        ----------
        params : list or tuple of str or str, optional
            Parameters to be reset. It should contain only valid parameters
            (`'alpha'`, `'beta'`, `'gamma'`, `'theta'` or `'W'`), or be a single
            parameter given as a string. Defaults to `None`, in which case all
            parameters are reset.

        Raises
        ------
        TypeError
            If `params` is neither a list, a tuple nor a string.
        PopNetError
            If an entry of `params` is not a valid population parameter.
        """
        valid_params = ('alpha', 'beta', 'gamma', 'theta', 'W')
        if params is None:
            params = valid_params
        if isinstance(params, str):
            params = (params,)
        if not isinstance(params, (list, tuple)):
            raise TypeError('\'params\' should be a list, tuple or string.')
        if any(param not in valid_params for param in params):
            raise PopNetError(f'An entry in {params} is not a valid population '
                              'parameter.')
        rng = np.random.default_rng()
        if 'alpha' in params:
            self._alpha = np.concatenate(
                        [rng.logistic(pop.alpha, pop.scale_alpha, size=pop.size)
                         for pop in self.populations])
        if 'beta' in params:
            self._beta  = np.concatenate(
                        [rng.logistic(pop.beta,  pop.scale_beta,  size=pop.size)
                         for pop in self.populations])
        if 'gamma' in params:
            self._gamma = np.concatenate(
                        [rng.logistic(pop.gamma, pop.scale_gamma, size=pop.size)
                         for pop in self.populations])
        if 'theta' in params:
            self._theta = np.concatenate(
                        [rng.logistic(pop.theta, pop.scale_theta, size=pop.size)
                         for pop in self.populations])
        if 'W' in params:
            self._W = np.block([[rng.logistic(
                                    self.c[J,K]/popK.size,
                                    self.scale_c[J,K]/popK.size,
                                    size=(popJ.size, popK.size)) 
                                for K, popK in enumerate(self.populations)] 
                                for J, popJ in enumerate(self.populations)])

    def size(self):
        """Get the size of the network."""
        return np.sum([pop.size for pop in self.populations])

    @property
    def underlying(self):
        raise AttributeError('\'MicroNetwork\' object has no attribute '
                             '\'underlying\'')


class Configuration:
    """Configurations used in numerical integrations and simulations.

    The `Configuration` class allows to easily group together all parameters
    that are needed to perform a numerical integrations of the dynamical system
    from the extended Wilson--Cowan model. Although the base class can be used
    with any network of any number of populations, it is preferable to use the
    `ConfigurationOne` subclass for the case where the network has only one
    population, and the `MicroConfiguration` subclass when the network has a
    defined microscopic structure.

    Parameters
    ----------
    network : Network
        Network associated to the configuration.
    ID : str, optional
        ID to associate to the configuration. Defaults to `None`, in which case
        the network's ID is used.
    **kwargs
        Keyword arguments used to initialize other data attributes.

    Attributes
    ----------
    ID : str
        ID of the configuration. See `Configuration.ID`.
    network : Network
        Network associated with the configuration. See `Configuration.network`.
    initial_state : array_like
        Initial state of the network. See `Configuration.initial_state`.
    Q : array_like
        Input in the network. See `Configuration.Q`.
    initial_time, final_time : float
        Times between which the evolution of the network's state is studied.
        See `Configuration.initial_time` and `Configuration.final_time`.
    iterations : int
        Number of iterations used for the numerical integration, if performed.
        See `Configuration.iterations`.
    delta : float
        Time interval between two iterations. See `Configuration.delta`.

    Raises
    ------
    TypeError
        If the second argument is not a `Network` instance.
    KeyError
        If a keyword argument is not a valid attribute.

    """

    def __init__(self, network, ID=None, **kwargs):
        if not isinstance(network, Network):
            raise TypeError('The network associated with a configuration '
                            'should indeed be a Network instance.')
        self._network = network

        if ID is None:
            ID = network.ID
        self.ID = ID

        state_attributes = {'Q': np.zeros(p := len(self.network.populations)), 
                            'initial_state': np.zeros(p * (2*p + 3))}
        float_attributes = {'initial_time': 0., 'final_time': 10.}
        int_attributes = {'iterations': 1000}

        # Here the time attributes have to be initialized without calling the 
        # setter methods, because they reference each other, so they all have to
        # be already defined when a setter is called. 
        for attr in kwargs:
            if attr in state_attributes:
                setattr(self, attr, kwargs[attr])
                state_attributes.pop(attr)
            elif attr in float_attributes:
                setattr(self, '_'+attr, float(kwargs[attr]))
                float_attributes.pop(attr)
            elif attr in int_attributes:
                setattr(self, '_'+attr, int(kwargs[attr]))
                int_attributes.pop(attr)
            else:
                raise KeyError(f'{attr} is not a valid parameter for the '
                               'Configuration class.')

        for attr in state_attributes:
            setattr(self, attr, state_attributes[attr])
        for attr in float_attributes:
            setattr(self, '_'+attr, float_attributes[attr])
        for attr in int_attributes:
            setattr(self, '_'+attr, int_attributes[attr])

        self.delta = (self.final_time - self.initial_time) / self.iterations

        A_labels = [f'A[{pop.ID}]' for pop in self.network.populations]
        R_labels = [f'R[{pop.ID}]' for pop in self.network.populations]
        CAA_labels = [f'CAA[{popJ.ID},{popK.ID}]'
                        for J, popJ in enumerate(self.network.populations)
                        for popK in self.network.populations[J:]]
        CRR_labels = [f'CRR[{popJ.ID},{popK.ID}]'
                        for J, popJ in enumerate(self.network.populations)
                        for popK in self.network.populations[J:]]
        CAR_labels = [f'CAR[{popJ.ID},{popK.ID}]'
                        for popJ in self.network.populations
                        for popK in self.network.populations]
        self._variables = (A_labels + R_labels 
                            + CAA_labels + CRR_labels + CAR_labels)

    def __str__(self):
        string = (f'Configuration {self.ID}\n\n'
                  f'Network used: {self.network.ID}\n')
        string += ('\nParameters:\n'
                  f'       ti = {self.initial_time}\n'
                  f'       tf = {self.final_time}\n'
                  f'       \u0394t = {self.delta}\n'
                  f'{self.iterations:>9} iterations\n')
        if self._other_params_string() is not None:
            string += self._other_params_string()
        string += ('\nInput:\n'
                  f'        Q = {self.Q}\n')
        string += '\nInitial state:\n'
        for var, val in zip(self._variables, self.initial_state):
            string += f'{var:>9} = {val}\n'
        return string[:-1] #Remove the last '\n'

    @staticmethod
    def create(network, ID=None, **kwargs):
        """Alias for `make_config`."""
        return make_config(network, ID=None, **kwargs)

    @staticmethod
    def load(load_ID, new_ID=None, network=None, folder=None):
        """Alias for `load_config`."""
        return load_config(load_ID, new_ID=None, network=None, folder=None)

    @property
    def ID(self):
        """ID of the configuration.

        Identificator given to the configuration. It has to be a string that
        begins with the associated network's ID. It is notably used to name
        files when saving the configuration. Setting its value will raise an
        error if it is not a string or if it does not begin with the network's
        ID.
        """
        return self._ID

    @ID.setter
    def ID(self, new_ID):
        if not isinstance(new_ID, str):
            raise TypeError('The configuration\'s ID should be a string.')
        if new_ID[:len(self.network.ID)] != self.network.ID:
            raise ValueError('The ID of the configuration should begin with '
                             'that of the network')
        self._ID = new_ID

    @property
    def network(self):
        """Network associated with the configuration.

        Network associated with the configuration, as a `Network` instance. It
        is set at initialization, but it cannot be reset nor deleted afterwards.
        """
        return self._network

    @property
    def initial_state(self):
        """Initial state of the configuration.

        The initial state of the network. As detailed in the
        [Notes](#configuration-initial-state-notes) section below, if the
        network has *p* populations, the initial state always has *p*(2*p*+3)
        components. The setter method ensures that the initial state is always
        a NumPy array of floats of the correct length. 

        Notes {#configuration-initial-state-notes}
        -----
        In the case where the network has *p* populations, the system has
        *p*(2*p*+3) dimensions: there are *p* equations for *A*'s, *p* equations
        for *R*'s, *p*(*p*+1)/2 equations for covariances between *A*'s,
        *p*(*p*+1)/2 equations for covariances between *R*'s, and
        *p*<sup>2</sup> equations for covariances between *A*'s and *R*'s. The
        states are assumed to be ordered as follows:
        \\[ 
            [\\begin{aligned}[t]
            & A_1, A_2, ..., A_p, \\\\
            & R_1, R_2, ..., R_p, \\\\
            & \\mathrm{C}_{AA}^{11}, \\mathrm{C}_{AA}^{12}, ..., 
                \\mathrm{C}_{AA}^{1p}, \\mathrm{C}_{AA}^{22}, ..., 
                \\mathrm{C}_{AA}^{2p}, \\mathrm{C}_{AA}^{33}, ..., 
                \\mathrm{C}_{AA}^{3p}, ..., \\mathrm{C}_{AA}^{pp}, \\\\
            & \\mathrm{C}_{RR}^{11}, \\mathrm{C}_{RR}^{12}, ..., 
                \\mathrm{C}_{RR}^{1p}, \\mathrm{C}_{RR}^{22}, ..., 
                \\mathrm{C}_{RR}^{2p}, \\mathrm{C}_{RR}^{33}, ..., 
                \\mathrm{C}_{RR}^{3p}, ..., \\mathrm{C}_{RR}^{pp}, \\\\
            & \\mathrm{C}_{AR}^{11}, \\mathrm{C}_{AR}^{12}, ..., 
                \\mathrm{C}_{AR}^{1p}, \\mathrm{C}_{AR}^{21}, ...,
                \\mathrm{C}_{AR}^{2p}, ..., \\mathrm{C}_{AR}^{p1}, 
                \\mathrm{C}_{AR}^{p2}, ..., \\mathrm{C}_{AR}^{pp}]
            \\end{aligned}
        \\]
        Remark that there are no \\(\\mathrm{C}_{AA}^{21}\\) or
        \\(\\mathrm{C}_{RR}^{21}\\) components, for example, since the
        \\(\\mathrm{C}_{AA}\\) and \\(\\mathrm{C}_{RR}\\) matrices are symmetric
        and each independant state variable is given only one. This is not the
        case for \\(\\mathrm{C}_{AR}\\), since
        \\[
            \\mathrm{C}_{AR}^{JK} = \\mathrm{Cov}[A_J, R_K] \\neq 
            \\mathrm{Cov}[A_K, R_J] = \\mathrm{C}_{AR}^{KJ} 
        \\]
        in general. 
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, new_state):
        length = (p := len(self.network.populations)) * (2*p + 3)
        if len(new_state) != length:
            raise ValueError(f'The state provided has {len(new_state)} '
                             f'components, but it should have {length} '
                             f'components for a network of {p} populations.')
        self._initial_state = np.array(new_state, float)

    @property
    def Q(self):
        """Input in the network.

        Input in the populations of the network from an external source. It must
        have the same length as the number of populations of the network. The
        setter method ensures that the input is always of the correct length,
        and that it is always a NumPy array of floats. 
        """
        return self._Q

    @Q.setter
    def Q(self, new_Q):
        try:
            float_new_Q = float(new_Q)
        except:
            pass
        else:
            new_Q = [float_new_Q]
        if len(new_Q) != (p := len(self.network.populations)):
            raise PopNetError(f'The input Q should always have {p} components '
                              f'for a network of {p} populations.')
        self._Q = np.array(new_Q, float)

    @property
    def initial_time(self):
        """Time from which the network's state is studied.

        Start of the period in which the evolution of the network's state is
        studied. When setting the initial time, the time interval `delta` is
        adapted to ensure that the number of iterations and the time interval
        are still consistent with the total duration of the integration.
        """
        return self._initial_time

    @initial_time.setter
    def initial_time(self, new_initial_time):
        self._initial_time = float(new_initial_time)
        self._delta = (self.final_time - self.initial_time) / self.iterations

    @property
    def final_time(self):
        """Time until which the network's state is studied.

        End of the period in which the evolution of the network's state is
        studied. When setting the final time, the time interval `delta` is
        adapted to ensure that the number of iterations and the time interval
        are still consistent with the total duration of the integration.
        """
        return self._final_time

    @final_time.setter
    def final_time(self, new_final_time):
        self._final_time = float(new_final_time)
        self._delta = (self.final_time - self.initial_time) / self.iterations

    @property
    def delta(self):
        """Time interval between two iterations in the numerical integration.
        
        Time interval between two consecutive iterations in a numerical
        integration performed using this configuration. This is not used for
        simulations of the microscopic network's dynamics. When setting the time
        interval, the number of iterations `iterations` is adapted to ensure
        that the number of iterations and the time interval are still consistent
        with the total length of time of the integration.
        """
        return self._delta

    @delta.setter
    def delta(self, new_delta):
        self._delta = float(new_delta)
        self._iterations = round((self.final_time - self.initial_time) 
                                 / self.delta)

    @property
    def iterations(self):
        """Number of iterations of the numerical integration.

        Total number of iterations of a numerical integration performed using
        this configuration. It is also the number of time steps added after the
        initial time to get the times array when doing statistics from
        simulations of the network's microscopic dynamics. In both cases, the
        length of the times array is `1 + iterations`. This is not used when
        only *one* simulation of the microscopic dynamics is performed.

        When setting the number of iterations, the time interval `delta` will be
        adapted to ensure that the number of iterations and the time interval
        are still consistent with the total length of time of the integration. 
        """
        return self._iterations

    @iterations.setter
    def iterations(self, new_number_of_iterations):
        self._iterations = int(new_number_of_iterations)
        self._delta = (self.final_time - self.initial_time) / self.iterations

    def add_random_uniform_perturbation(self, R, indices=None):
        """Add a random perturbation to the initial state.

        Add a random perturbation to the initial state, taken from a uniform
        distribution on an *N*--sphere of radius `R`. The dimension *N* is the
        nunmber of components perturbated, described by `indices`. 

        Parameters
        ----------
        R : float
            Norm of the perturbation. Corresponds to the radius of the
            *N*--sphere in which the perturbation is randomly taken.
        indices : list or tuple of ints, optional
            Indices giving the components to change in the initial state.
            Defaults to `None`, in which case every component is changed. 

        Notes
        -----
        To generate a uniform distribution on an *N*--sphere of radius `R`, we
        use the method described in [1]: every component of the perturbation is
        first taken from a standard normal distribution, and then the resulting
        vector is scaled to have a norm of `R`.

        References
        ----------
         1. Muller, M. E. A note on a method for generating points uniformly on
            *N*-dimensional spheres. *Commun. ACM* **2**, 19--20 (1959).
            doi:[10.1145/377939.377946](https://doi.org/10.1145/377939.377946).
        """
        if indices is None:
            indices = np.arange(len(self.initial_state))
        ball = np.random.default_rng().normal(size=len(indices))
        ball = R * ball / np.linalg.norm(ball)
        perturbation = np.zeros(len(self.initial_state))
        perturbation[np.array(indices)] = ball
        self.initial_state = self.initial_state + perturbation

    def add_to_initial_state(self, perturbation):
        """Add a given perturbation to the initial state."""
        if perturbation is None:
            return 
        self.initial_state = self.initial_state + np.array(perturbation, float)

    def copy(self, new_ID):
        """Copy the configuration.

        Return a copy of the configuration with a new ID.

        Parameters
        ----------
        new_ID : str
            ID to give to the new configuration.

        Returns
        -------
        Configuration
            The copied configuration.
        """
        new_config = deepcopy(self)
        new_config.ID = new_ID
        return new_config

    def save(self, folder=None, note=None):
        """Save the current configuration.

        Save the string representation of the configuration in a text file,
        under the name *ID - Configuration.txt*, where *ID* is the actual ID of
        the configuration.

        Parameters
        ----------
        folder : str, optional
            If not `None`, the file will be saved in this folder. It should
            already exist in the current directory. Defaults to `None`, in which
            case the file will be saved directly in the current directory.
        note : str, optional
            If not `None`, an additional section "Additional notes:" will be
            written in the file, and `note` will be written there. 
        """
        filename = _format_filename(folder, self.ID, 'Configuration')
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(self))
            if note is not None:
                file.write('\n\nAdditional notes:\n')
                file.write(note)

    def set_random_initial_state(self, bound_cov=0.06):
        """Set the initial state with random values.

        Set the initial state with random values. For each population, the
        values for *A* and *R* and chosen from uniform distributions in the
        triangle \\(\\{(x,y) \\in [0,1)^2 : x + y < 1\\}\\), using the method
        described in [1]. All variances are chosen from uniform distributions
        between 0 and `bound_cov`, and all non-symmetric covariances from
        uniform distributions between `-bound_cov` and `bound_cov`, regardless
        of the values of the expectations. 

        Parameters
        ----------
        bound_cov : float, optional
            Positive number which sets the distributions of covariances.
            Variances are all taken from a uniform distribution between zero and
            `bound_cov`, and non-symmetric covariances are taken from a uniform
            distribution between `-bound_cov` and `bound_cov`. Defaults to 0.06.

        Raises
        ------
        TypeError
            If `bound_cov` cannot be converted to a float.
        ValueError
            If `bound_cov` is not a positive number.

        References
        ----------
         1. Osada, R., Funkhouser, T., Chazelle, B. & Dobkin, D. Shape
            distributions. *ACM Trans. Graph.* **21**, 807--832 (2002).
            doi:[10.1145/571647.571648](https://doi.org/10.1145/571647.571648).
        """
        p = len(self.network.populations)
        state = np.zeros(p*(2*p+3))
        rng = np.random.default_rng()
        try:
            bound_cov = float(bound_cov)
        except TypeError as error:
            raise TypeError('The bound to choose covariances should be a '
                            'number.') from error
        if bound_cov < 0:
            raise ValueError('The bound to choose covariances should be '
                             'positive.')
        for J in range(p):
            a, b = rng.random(size=2)
            state[J] = np.sqrt(a) * (1 - b)
            state[p+J] = np.sqrt(a) * b
        state[2*p : p*(p+3)] = bound_cov * rng.random(size=p*(p+1))
        state[p*(p+3) :] = -bound_cov + 2*bound_cov * rng.random(size=p**2)
        self.initial_state = state

    def _other_params_string(self):
        """Optional parameters for subclasses to write in `str(self)`."""
        pass


class MicroConfiguration(Configuration):
    """Configurations used in numerical simulations.

    The `MicroConfiguration` class extends the `Configuration` class to cases
    where the microscopic structure of the network is needed to perform
    numerical simulations. It adds two new properties:

     - `MicroConfiguration.micro_initial_state`, which gives the network's
       microscopic initial state;
     - `MicroConfiguration.executions`, which gives the number of trajectories
       generated when performing simulations in chain.

    It also provides a method to reset it from the macroscopic initial state,
    and another to modify the sizes of the populations of the network while also
    updating the network's parameters and the microscopic initial state.

    Since this configuration class requires the network to have a microscopic
    structure, it has to be initialized from a `MicroNetwork` instance. Besides
    that, the initialization is the same as in the base class.

    The data attributes are the same as in the base class.

    Raises
    ------
    PopNetError
        If the network used is not a `MicroNetwork`.

    """

    def __init__(self, network, ID=None, **kwargs):
        if 'executions' in kwargs:
            self.executions = kwargs.pop('executions')
        else:
            self.executions = 1
        super().__init__(network, ID=ID, **kwargs)
        if not isinstance(network, MicroNetwork):
            raise PopNetError('The network used with a MicroConfiguration '
                              'should be a MicroNetwork.')

    @Configuration.initial_state.setter
    def initial_state(self, new_state):
        Configuration.initial_state.fset(self, new_state)
        self.reset_micro_initial_state()

    @property
    def micro_initial_state(self):
        """Microscopic initial state of the network.

        States of all neurons of the network. It is always consistent with the
        macrosopic initial state `Configuration.initial_state`, in the sense
        that the microscopic initial state can only be set from macroscopic one.
        For more details on this process, see
        `MicroConfiguration.reset_micro_initial_state`.

        The microscopic initial state cannot be set manually, but it can be
        reset at any time with `MicroConfiguration.reset_micro_initial_state`.
        Note also that when the macroscopic initial state is changed, the
        microscopic one is also reset.
        """
        return self._micro_initial_state

    @property
    def executions(self):
        """Number of simulations to be done.

        Number of simulations to be performed when doing simulations in chain
        to obtain statistics. It cannot be deleted.
        """
        return self._executions

    @executions.setter
    def executions(self, new_number):
        try:
            new_number = int(new_number)
        except TypeError as error:
            raise TypeError('The number of simulations to be done should be '
                            'a number.') from error
        if new_number < 0:
            raise ValueError('The number of simulations to be done has to be '
                             'positive.')
        self._executions = new_number

    def resize_network(self, new_sizes):
        """Change the sizes of the network's populations.

        Change the size of each population of the network, and reset the
        network's parameters and the microscopic initial state to be consistent
        with this change.

        Parameters
        ----------
        new_sizes : list or tuple of int
            A new size for each population, given in the order prescribed by
            `network.populations`, where `network` is the attribute.
        """
        for pop, new_size in zip(self.network.populations, new_sizes):
            pop.size = new_size
        self.network.reset_parameters()
        self.reset_micro_initial_state()
        
    def reset_micro_initial_state(self):
        """Randomly generate a microscopic initial state.
        
        Create a microscopic initial state for the network, consistent with its
        macroscopic initial state. If *J* is a population of the network, each
        neuron of *J* is chosen randomly between the values `1` (active), `1j`
        (refractory) and `0` (sensitive), with probabilities corresponding to
        the active, refractory and sensitive fractions of *J*.
        """
        A = self.initial_state[: (p := len(self.network.populations))]
        R = self.initial_state[p : 2*p]
        S = 1 - A - R
        rng = np.random.default_rng()
        self._micro_initial_state = np.concatenate(
                [rng.choice((0.,1.,1j), p=(S[J],A[J],R[J]), size=popJ.size)
                 for J, popJ in enumerate(self.network.populations)])

    def _other_params_string(self):
        """Add `executions` to `str(self)`."""
        if self.executions == 1:
            return f'{1:>9} execution\n'
        else:
            return f'{self.executions:>9} executions\n'


class ConfigurationOne(Configuration):
    """Extends `Configuration` in the special case of a single population.

    Extends the `Configuration` class by adding methods specific to the case of 
    only one population. The new methods allow to:

     - Verify if a state is in the domain where variables make sense,
       physiologically speaking;
     - Set the initial state randomly in the physiological domain;
     - Set the input and the initial state to coordinates where there is a fixed
       point.

    The initialization is the same as in the base class, except for a
    verification that the network has indeed a single population.

    The data attributes are the same as in the base class.

    Raises
    ------
    PopNetError
        If the network does not have precisely one population.

    """

    def __init__(self, network, ID=None, **kwargs):
        super().__init__(network, ID=ID, **kwargs)
        if network.ID[0] != '1':
            raise PopNetError('The subclass ConfigurationOne should be used '
                              'only for configurations where the network has '
                              'indeed one population. The network used here '
                              f'has {network.ID[0]}')
        self._variables = ['A', 'R', 'CAA', 'CRR', 'CAR']

    def set_random_initial_state(self, domain='physiological'):
        """Set the initial state randomly.

        Overrides the corresponding base class method to choose an initial state
        in a given domain. The expectations are always chosen from a uniform
        distribution in the triangle \\(\\{(x,y) \\in [0,1)^2 : x + y < 1\\}\\).

        Parameters
        ----------
        domain : {'physiological', 'bounded'}, optional
            The domain in which the state is chosen. If 'physiological', the
            state is chosen in the so-called physiological domain, where
            expectations and covariances are valid values for the random
            variables they represent. If 'bounded', the base class method is
            called. Defaults to `'physiological'`.

        Raises
        ------
        NotImplementedError
            If the requested domain is not valid. 

        See Also
        --------
        Configuration.set_random_initial_state
        """
        rng = np.random.default_rng()
        if domain == 'physiological':
            a, b = rng.random(size=2)
            A = np.sqrt(a) * (1 - b)
            R = np.sqrt(a) * b
            CAA = A * (1 - A) * rng.random()
            CRR = R * (1 - R) * rng.random()
            CAR = -np.sqrt(CAA*CRR) + 2*np.sqrt(CAA*CRR) * rng.random()
            new_state = [A, R, CAA, CRR, CAR]
            if self.state_in_domain(new_state, verbose=False):
                self.initial_state = new_state
            else:
                self.set_random_initial_state()
            return
        elif domain == 'bounded':
            super().set_random_initial_state()
            return
        raise NotImplementedError(f'No "{domain}" domain has been implemented '
                                  'yet for the states of a network with a '
                                  'single population.')

    def set_to_fixed_point(self, kind, set_state=True):
        """Place the system at a fixed point.

        Set the input and (possibly) the initial state to coordinates where
        there is a fixed point of type *i*), *ii*) or *iii*). It is possible to
        set only the input and not the intial state with the `set_state`
        argument.

        Parameters
        ----------
        kind : {'i', 'ii', 'iii'}
            The desired type of fixed point.
        set_state : bool, optional
            Decides if the initial state is set to or near the fixed point.
            Defaults to `True`.

        Raises
        ------
        NotImplementedError
            If `kind` is `'i'`. 
        ValueError
            If `kind` is neither of `'i'`, `'ii'` or `'iii'`.
        """
        pop = self.network.populations[0]
        c = self.network.c[0,0]
        Sigma = pop.alpha/2 + pop.beta + pop.gamma
        Pi = pop.alpha*pop.beta/2 + pop.alpha*pop.gamma/2 + pop.beta*pop.gamma
        if kind == 'i':
            raise NotImplementedError('The coordinates for a type i fixed point'
                                      ' have not been implemented yet.')
        elif kind == 'ii':
            S = 4 * pop.scale_theta * Sigma / (pop.alpha * c)
            A = pop.gamma / (pop.beta + pop.gamma) * (1 - S)
            R = pop.beta  / (pop.beta + pop.gamma) * (1 - S)
            CAR = (2*pop.gamma / (pop.beta+pop.gamma) * (Sigma * Pi - pop.beta
                    * pop.gamma * pop.alpha * c / (4*pop.scale_theta))
                    / ( (pop.alpha + 2*pop.gamma) * 
                        (pop.alpha * c / (4*pop.scale_theta))**2 ))
            CAA = pop.alpha / (2*pop.gamma) * CAR
            CRR = pop.beta / pop.gamma * CAR
            self.Q = pop.theta - c * A
        elif kind == 'iii':
            S = 4 * pop.scale_theta * Pi / (pop.gamma * pop.alpha * c)
            A = pop.gamma / (pop.beta + pop.gamma) * (1 - S)
            R = pop.beta  / (pop.beta + pop.gamma) * (1 - S)
            CAR = (pop.beta / pop.gamma * ( Pi**2 - pop.beta*pop.gamma**2
                    * pop.alpha * c / (4*pop.scale_theta) )
                    / ( (pop.beta+pop.gamma)**2 
                        * (pop.alpha * c / (4*pop.scale_theta))**2 ))
            CAA = pop.gamma / pop.beta * CAR
            CRR = pop.beta / pop.gamma * CAR
            self.Q = pop.theta - c * A
        else:
            raise ValueError('Unknown fixed point type.')
        if set_state:
            self.initial_state = [A, R, CAA, CRR, CAR]

    def state_in_domain(self, state=None, verbose=False):
        """Verify if a state is in the physiological domain.

        Parameters
        ----------
        state : array_like, optional
            The state to verify. Defaults to `None`, in which case the initial
            state is verified.
        verbose : bool, optional
            If `True`, a warning will be issued if the state is not in the
            physiological domain. Defaults to `False`.

        Returns
        -------
        bool
            `True` if the state is in the physiological domain, else `False`.

        Warns
        -----
        PopNetWarning
            If `verbose` is `True` and if the state is not in the physiological
            domain.
        """
        if state is None:
            state = self.initial_state
        A = state[0]
        R = state[1]
        S = 1 - A - R
        CAA = state[2]
        CRR = state[3]
        CAR = state[4]
        CSS = CAA + 2*CAR + CRR
        if A < 0 or R < 0 or S < 0:
            ok = False
        elif CAA > A * (1 - A) or CAA < 0:
            ok = False
        elif CRR > R * (1 - R) or CRR < 0:
            ok = False
        elif CSS > S * (1 - S) or CSS < 0:
            ok = False
        elif CAR**2 > CAA * CRR:
            ok = False
        elif CAR < -A*R:
            ok = False
        elif CAR > A*S - CAA:
            ok = False
        elif CAR > R*S - CRR:
            ok = False
        else:
            ok = True
        if not ok and verbose:
            warn(f'The state {state} does not make sense, physiologically '
                 'speaking.', category=PopNetWarning, stacklevel=2)
        return ok


class MicroConfigurationOne(ConfigurationOne, MicroConfiguration):
    """Extends `MicroConfiguration` in the special case of a single population.

    Combines the features of `ConfigurationOne` and `MicroConfiguration` in
    order to be used for cases where the microscopic structure of a network
    containing a single population is needed.

    """

    pass


class Executor:
    """Execute numerical experiments on a network.

    The `Executor` class is meant to perform numerical experiments to study the
    dynamics of a network split into populations. These experiments are intended
    to be carried out by subclasses of `Executor`.

     - To perform simulations of the stochastic process which rules the
       evolution of the network, use `Simulator`.
     - To perform numerical integrations of reduced dynamical systems
       describing the macroscopic behavior of the network, use `Integrator`.

    Parameters
    ----------
    config : Configuration
        Configuration used for the experiments.

    Attributes
    ----------
    config : Configuration
        Configuration used for the experiments. See `Executor.config`.
    times : array_like
        Time. See `Executor.times`.
    states : array_like
        State of the network with respect to time. See `Executor.states`.

    """

    def __init__(self, config):
        self.config = config

    @property
    def config(self):
        """Configuration used with the executor.

        Configuration defining all parameters used by the executor. It must be a
        `Configuration` instance, or a `MicroConfiguration` instance if the
        network should have a microscopic structure. If it is set, the executor
        is reset with `Executor.reset`. It cannot be deleted.
        """
        return self._config

    @config.setter
    def config(self, new_value):
        if not isinstance(new_value, Configuration):
            raise TypeError('The configuration used with an executor must be a '
                            'Configuration instance.')
        self._config = new_value
        self.reset()

    @property
    def states(self):
        """State of the network with respect to time.

        Macroscopic state of the network at each time step. It does not contain
        any relevant data at initialization or right after a reset, but it is
        updated during a call to `run`. It cannot be manually set nor deleted.
        """
        return self._states

    @property
    def times(self):
        """Time.

        At initialization or with a call to `Executor.reset`, it is set
        according to the integrator's configuration `config`. Specifically, it
        is an array starting at `config.initial_time` and ending at
        `config.final_time`, with an interval of `config.delta` between time
        steps. It cannot be manually set nor deleted.
        """
        return self._times

    def close(self):
        """Delete all data attributes of the executor."""
        del self._config
        del self._states
        del self._times
        del self._success

    def output(self, **kwargs):
        """Get the output of the execution.

        Return the results of the numerical experiment.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the output's class constructor.

        Returns
        -------
        Result
            The output of the experiment. The precise output type depends on the
            experiment executed and on the number of populations of the network.
            See the [summary](#classes-and-hierarchy) of the module's classes
            for a quick reference giving the output type of each executor class.

        Raises
        ------
        PopNetError
            If the numerical experiment has not been performed yet.
        """
        self._check_if_run()
        ResultClass = self._output_type()
        return ResultClass(self.config, self._output_states(), 
                           self._output_times(), **kwargs)

    def reset(self):
        """Reset the executor."""
        self._success = None
        self._times = np.linspace(self.config.initial_time, 
                                  self.config.final_time, 
                                  1 + self.config.iterations)

    def run(self):
        raise NotImplementedError('An Executor must implement a \'run\''
                                  'method.')

    def save_output(self, name=None, folder=None):
        """Save the output of the experiment in a text file.

        Save the output of the numerical experiment in a text file, under the
        name *ID - name.txt*, where *ID* is the ID of the configuration used
        for the experiment and *name* is `name`.

        Parameters
        ----------
        name : str, optional
            Name to give to the saved output. Defaults to `None`, in which case
            it is replaced with a default based on the output class.
        folder : str, optional
            If given, the file will be saved in this folder. It should already
            exist in the current directory. Defaults to `None`, in which case
            the file will be saved directly in the current directory.

        Returns
        -------
        name : str
            The name given to the saved output.

        Raises
        ------
        PopNetError
            If the numerical experiment has not been performed yet.
        """
        self._check_if_run()
        name = self._output_type()._get_name(name)
        filename = _format_filename(folder, self.config.ID, name)
        L = self._state_length()
        header = ''.join([f'{X:<16}' for X in self.config._variables[:L]])
        np.savetxt(filename, self._output_states(), fmt='%+.12f', header=header)
        return name

    def _check_if_run(self):
        """Check if the executor has already run."""
        if self._success is None:
            raise PopNetError('An Executor should run before the results are '
                              'output. Call run() first.')

    def _output_states(self):
        """States array to output."""
        return self.states

    def _output_times(self):
        """Times array to output."""
        return self.times

    @classmethod
    def _output_type(cls):
        """Type of output."""
        raise NotImplementedError('An Executor must give an output type.')

    def _state_length(self):
        """Length of the macroscopic states."""
        raise NotImplementedError('An Executor must give its states\' sizes.')


class Integrator(Executor):
    """Numerical integrator for ODEs related to Wilson--Cowan's model.

    The `Integrator` class extends the `Executor` class to perform numerical
    integrations of dynamical systems related to the Wilson--Cowan model.
    All numerical integrations are performed with the class
    [ode](https://tinyurl.com/scipy-integrate-ode) from SciPy's `integrate`
    module. Specific vector fields are implemented in `Integrator`'s subclasses.

    Parameters
    ----------
    config : Configuration
        Sets the configuration to use for the integration. 

    Attributes
    ----------
    config : Configuration
        The configuration used for the numerical integration. See
        `Integrator.config`.
    states, times : array_like
        Arrays representing the state of the network with respect to time. See
        `Integrator.states` and `Integrator.times`.

    """

    @staticmethod
    def create(config, **kwargs):
        """Alias for `get_integrator`."""
        return get_integrator(config, **kwargs)

    def reset(self):
        """Reset the integrator."""
        super().reset()
        self._states = np.zeros((len(self.times), self._state_length()))
        self.states[0] = self.config.initial_state[: self._state_length()]

    def run(self, backend='vode', success_test='ode', verbose=True):
        """Run the numerical integration.

        Run a numerical integration of the dynamical system using an
        [ode](https://tinyurl.com/scipy-integrate-ode) instance from SciPy's
        `integrate` module.

        Parameters
        ----------
        backend : {'vode', 'zvode', 'lsoda', 'dopri5', 'dop853'}, optional
            Integrator used with `ode`. Defaults to `'vode'`.
        success_test : {'ode', 'domain'}, optional
            Success test. If it fails at an iteration, the integration stops.
            If `'ode'`, the test is that given by `ode`. If
            `'domain'`, the integration is considered to have failed whenever
            a commponent gets greater than 1. Defaults to `'ode'`.
        verbose : bool, optional
            Issue a warning if the integration fails. Defaults to `True`.

        Warns
        -----
        PopNetWarning
            If the integration fails and if `verbose` is `True`.
        """
        if self._jac.__qualname__.startswith('Integrator'):
            solver = ode(self._field)
        else:
            solver = ode(self._field, self._jac)
        solver.set_integrator(backend)
        solver.set_initial_value(self.states[0])
        self._success = True
        if success_test == 'ode':
            def test(solver, state): return solver.successful()
        elif success_test == 'domain':
            def test(solver, state): return max(state) < 1
        else:
            raise ValueError(f'No test {success_test} known to Integrator.')
        for j in range(1, len(self.times)):
            self.states[j] = solver.integrate(solver.t+self.config.delta)[:]
            if not test(solver, self.states[j]):
                self._success = False
                break
        if not self._success and verbose:
            warn(f'Integration failed with configuration {self.config.ID}.',
                 category=PopNetWarning, stacklevel=2)

    def _field(self, t, Y):
        """Vector field.

        Vector field corresponding to the studied dynamical system.

        Parameters
        ----------
        t : float
            Current time.
        Y : array_like
            Current state of the network.

        Returns
        -------
        array_like
            Gradient of the vector field evaluated at time `t` and state `Y`.
        """
        raise NotImplementedError('An integrator must implement a vector field.')

    def _jac(self, t, Y):
        """Jacobian of the `_field` method.

        Jacobian of the vector field corresponding to the studied dynamical
        system. It does not have to be implemented by subclasses.

        Parameters
        ----------
        t : float
            Current time.
        Y : array_like
            Current state of the network.

        Returns
        -------
        array_like
            Jacobian of the vector field evaluated at time `t` and state `Y`.
        """
        pass

    @staticmethod
    def _unflat_triangle(Y):
        """Reshape a flatten symmetric matrix into a square one.

        Reshape a flatten symmetric matrix from a one-dimensional array which
        contains only the upper triangle of the symmetric square matrix. It is
        mainly intended to be used internally to convert parts of state vectors
        into square matrices representing covariances.

        Parameters
        ----------
        Y : array_like
            The one-dimensional array to reshape into a square array.

        Returns
        -------
        array_like
            The array correctly reshaped.
        """
        new_array = np.zeros((p := round((-1 + np.sqrt(1 + 8*len(Y))) / 2), p))
        new_array[np.triu_indices(p)] = Y
        new_array[np.tril_indices(p, k=-1)] = new_array[np.triu_indices(p, k=1)]
        return new_array


class SimpleIntegrator(Integrator):
    """Integrator for the Wilson--Cowan model.

    Specializes the `Integrator` class for the Wilson--Cowan model with the
    refractory state included. Covariances are not considered in this case.

    """

    def _field(self, t, Y):
        """Vector field of the Wilson--Cowan model. See `Integrator._field`."""
        A = Y[: (p := len(self.config.network.populations))]
        R = Y[p :]
        S = 1 - A - R
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        f = np.zeros(2*p)
        for J, popJ in enumerate(self.config.network.populations):
            f[J] = - popJ.beta * A[J] + popJ.alpha*popJ.F(B[J]) * S[J]
            f[J+p] = - popJ.gamma * R[J] + popJ.beta * A[J]
        return np.array(f, float)

    def _jac(self, t, Y):
        """Jacobian of the vector field from the Wilson--Cowan model. See
        `Integrator._jac`."""
        A = Y[: (p := len(self.config.network.populations))]
        R = Y[p : 2*p]
        S = 1 - A - R
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        j = np.zeros((2*p, 2*p))
        for J, popJ in enumerate(self.config.network.populations):
            j[J,J] = (- popJ.beta - popJ.alpha*popJ.F(B[J]) + popJ.alpha
                        * popJ.dF(B[J])*self.config.network.c[J,J]*S[J] )
            j[J,J+p] = - popJ.alpha*popJ.F(B[J])
            j[J+p,J] = popJ.beta
            j[J+p,J+p] = - popJ.gamma
            for K, popK in enumerate(self.config.network.populations):
                j[J,K] = popJ.alpha*popJ.dF(B[J])*self.config.network.c[J,K]*S[J]
        return np.array(j, float)

    def _output_type(self):
        """Type of integration output."""
        if len(self.config.network.populations) == 1:
            return MeanFieldOne
        return MeanField

    def _state_length(self):
        """Length of the field's states."""
        return 2 * len(self.config.network.populations)


class ExtendedIntegrator(Integrator):
    """Integrator for the extended Wilson--Cowan model.

    Specializes the `Integrator` class for the extended Wilson--Cowan model,
    where the refractory state and covariances between fractions of populations
    are included.

    """

    def _field(self, t, Y):
        """Vector field of the extended Wilson--Cowan model. See
        `Integrator._field`."""
        A = Y[: (p := len(self.config.network.populations))]
        R = Y[p : 2*p]
        S = 1 - A - R
        CAA = self._unflat_triangle(Y[2*p : 2*p + round(p*(p+1)/2)])
        CRR = self._unflat_triangle(Y[2*p + round(p*(p+1)/2) : 2*p+p*(p+1)])
        CAR = (Y[2*p + p*(p+1) :]).reshape((p,p))
        CAS = - CAA - CAR
        CSR = - CRR - CAR
        B = self.config.Q.copy()
        VarB = np.zeros(p)
        CAB = np.zeros((p,p))
        CRB = np.zeros((p,p))
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
            for L in range(p):
                VarB[J] += (self.config.network.c[J,K] 
                            * self.config.network.c[J,L] * CAA[K,L])
                CAB[J,K] += self.config.network.c[K,L] * CAA[J,L]
                CRB[J,K] += self.config.network.c[K,L] * CAR[L,J]
        f = np.zeros(p*(2*p+3))
        dCAA = np.zeros((p,p))
        dCRR = np.zeros((p,p))
        dCAR = np.zeros((p,p))
        for J, popJ in enumerate(self.config.network.populations):
            f[J] = (- popJ.beta * A[J] + popJ.alpha*popJ.F(B[J]) * S[J]
                    - popJ.alpha*popJ.dF(B[J]) * (CAB[J,J] + CRB[J,J])
                    + popJ.alpha/2*popJ.ddF(B[J]) * S[J] * VarB[J])
            f[J+p] = - popJ.gamma * R[J] + popJ.beta * A[J]
            for K, popK in enumerate(self.config.network.populations):
                dCAA[J,K] = (- (popJ.beta + popK.beta) * CAA[J,K]
                                + popJ.alpha*popJ.F(B[J]) * CAS[K,J]
                                + popK.alpha*popK.F(B[K]) * CAS[J,K]
                                + popJ.alpha*popJ.dF(B[J]) * S[J] * CAB[K,J]
                                + popK.alpha*popK.dF(B[K]) * S[K] * CAB[J,K])
                dCRR[J,K] = (- (popJ.gamma + popK.gamma) * CRR[J,K]
                                + popJ.beta * CAR[J,K] + popK.beta * CAR[K,J])
                dCAR[J,K] = (- (popJ.beta + popK.gamma) * CAR[J,K]
                                + popK.beta * CAA[J,K]
                                + popJ.alpha*popJ.F(B[J]) * CSR[J,K]
                                + popJ.alpha*popJ.dF(B[J]) * S[J] * CRB[K,J])
        f[2*p : 2*p + round(p*(p+1)/2)] = dCAA[np.triu_indices(p)]
        f[2*p + round(p*(p+1)/2) : 2*p + p*(p+1)] = dCRR[np.triu_indices(p)]
        f[2*p + p*(p+1) :] = dCAR.flatten()
        return np.array(f, float)

    @classmethod
    def _output_type(cls):
        """Type of integration output."""
        return Solution

    def _state_length(self):
        """Length of the field's states."""
        p = len(self.config.network.populations)
        return p * (2*p + 3)


class ExtendedIntegratorOne(Integrator):
    """Integrator for the extended Wilson--Cowan model.

    Specializes the `Integrator` class for the extended Wilson--Cowan model,
    where the refractory state and covariances between fractions of populations
    are included, for the special case where the network has a single
    population. It is different from the `ExtendedIntegrator` class in that it
    uses a simpler implementation of the vector field.

    Raises
    ------
    PopNetError
        If the network does not have precisely one population.

    """

    def __init__(self, config):
        super().__init__(config)
        if config.ID[0] != '1':
            raise PopNetError('The subclass ExtendedIntegratorOne should be '
                              'used only for configurations where the network '
                              'has indeed one population. The network used here'
                              f' has {config.ID[0]}.')
        self._pop = self.config.network.populations[0]

    def _field(self, t, Y):
        """Vector field from the extended Wilson--Cowan model. See
        `Integrator._field`."""
        A, R, CAA, CRR, CAR = Y[0], Y[1], Y[2], Y[3], Y[4]
        S = 1 - A - R
        c = self.config.network.c[0,0]
        B = c * A + self.config.Q[0]
        F = self._pop.F(B)
        dF = self._pop.dF(B)
        ddF = self._pop.ddF(B)
        f = [0, 0, 0, 0, 0]
        f[0] = (- self._pop.beta*A + self._pop.alpha*F * S 
                - self._pop.alpha*dF * c * (CAA + CAR)
                + self._pop.alpha/2*ddF * c**2 * S * CAA )
        f[1] = - self._pop.gamma*R + self._pop.beta*A
        f[2] = (- 2*(self._pop.beta + self._pop.alpha*F) * CAA 
                - 2*self._pop.alpha*F*CAR 
                + 2*self._pop.alpha*dF * c* S * CAA )
        f[3] = - 2*self._pop.gamma*CRR + 2*self._pop.beta*CAR
        f[4] = (- (self._pop.beta + self._pop.gamma 
                + self._pop.alpha*F) * CAR + self._pop.beta*CAA 
                - self._pop.alpha*F*CRR 
                + self._pop.alpha*dF * c * S * CAR )
        return np.array(f, float)

    def _jac(self, t, Y):
        """Jacobian of vector field of the extended Wilson--Cowan model. See
        `Integrator._jac`."""
        A, R, CAA, CRR, CAR = Y[0], Y[1], Y[2], Y[3], Y[4]
        S = 1 - A - R
        c = self.config.network.c[0,0]
        B = c * A + self.config.Q[0]
        F = self._pop.F(B)
        dF = self._pop.dF(B)
        ddF = self._pop.ddF(B)
        dddF = self._pop.dddF(B)
        j = np.zeros((5,5))
        j[0,0] = (-self._pop.beta - self._pop.alpha*F 
                    + self._pop.alpha*dF * c * S 
                    - self._pop.alpha*ddF * c**2 * (CAA + CAR) 
                    + self._pop.alpha/2 * (-ddF + S*dddF*c) * c**2 * CAA )
        j[0,1] = (-self._pop.alpha*F 
                    - self._pop.alpha/2*ddF * c**2*CAA )
        j[0,2] = (-self._pop.alpha/2*ddF * c**2
                    - self._pop.alpha*dF * c )
        j[0,3] = 0
        j[0,4] = -self._pop.alpha*dF * c
        j[1,0] = self._pop.beta
        j[1,1] = -self._pop.gamma
        j[1,2] = j[1,3] = j[1,4] = 0
        j[2,0] = (-2*self._pop.alpha*dF*c * CAA 
                    - 2*self._pop.alpha*dF*c * CAR 
                    + 2*self._pop.alpha * (-dF + ddF*c*S) * c * CAA )
        j[2,1] = -2*self._pop.alpha*dF * c * CAA
        j[2,2] = (-2 * (self._pop.beta + self._pop.alpha*F) 
                    + 2*self._pop.alpha*dF * c * S )
        j[2,3] = 0
        j[2,4] = -2*self._pop.alpha*F
        j[3,0] = j[3,1] = j[3,2] = 0
        j[3,3] = -2*self._pop.gamma
        j[3,4] = 2*self._pop.beta
        j[4,0] = (-self._pop.alpha*dF * c * CAR 
                    - self._pop.alpha*dF * c * CRR 
                    + self._pop.alpha * (-dF + ddF*c*S) * c * CAR )
        j[4,1] = -self._pop.alpha*dF*c * CAR
        j[4,2] = self._pop.beta
        j[4,3] = -self._pop.alpha*F
        j[4,4] = (-(self._pop.beta + self._pop.gamma + self._pop.alpha*F) 
                    + self._pop.alpha*dF* c * S )
        return np.array(j, float)

    @classmethod
    def _output_type(cls):
        """Type of integration output."""
        return SolutionOne

    @classmethod
    def _state_length(self):
        """Length of the field's states."""
        return 5


class Simulator(Executor):
    """Numerical simulator of the stochastic process on a network.

    The `Simulator` class extends the `Executor` class to perform numerical
    simualtions of a stochastic process on a network whose mean field reduction
    is represented macroscopically by the Wilson--Cowan model.

    Parameters
    ----------
    config : MicroConfiguration
        Sets the configuration used for the simulation.

    Attributes
    ----------
    config : MicroConfiguration
        Configuration used for the simulations. See `Simulator.config`.
    states, times : array_like
        Macroscopic state of the network with respect to time. See
        `Simulator.states` and `Simulator.times`.
    micro_states, transition_times : array_like
        Microscopic state of the network with respect to time. See
        `Simulator.micro_states` and `Simulator.transition_times`.
    activation_rates : list
        Activation rates of the neurons of the network. See
        `Simulator.activation_rates`.

    """

    @Executor.config.setter
    def config(self, new_value):
        if not isinstance(new_value, MicroConfiguration):
            raise TypeError('The configuration used with a simulator must be a '
                            'MicroConfiguration instance.')
        self._config = new_value
        self.reset()

    @property
    def micro_states(self):
        """Microscopic state of the network with respect to time.

        Microscopic state of the network at each time step given by
        `Simulator.transition_times`. It does not contain any relevant data at
        initialization or right after a reset, but it is updated during a call
        to `run`. It cannot be manually set nor deleted.
        """
        return self._micro_states

    @property
    def transition_times(self):
        """Time.

        Times at which transitions have occurred for a given trajectory. Unlike
        `Simulator.times`, it is not set according to the configuration used,
        but rather updated stochastically during a call to `run`. It does not
        contain any relevant data at initialization or right after a reset. It
        cannot be manually set nor deleted.
        """
        return self._transition_times

    @property
    def activation_rates(self):
        """Activation rates of the neurons.

        List of functions representing the activation rates of the network's
        neurons. `activation_rates[j](x)` gives the activation rate of the
        *j*th neuron of the network if the state of the whole network is `x`.
        It cannot be manually set nor deleted.
        """
        return self._activation_rates

    def calcium_output(self, indices=None, growth_rate=None, decay_rate=None):
        """Get the calcium concentration in neural cells.

        Get the concentration of calcium in neural cells with respect to time. 

        Parameters
        ----------
        indices : int or array_like, optional
            Indices of neurons for which to get the calcium concentration.
            Defaults to `None`, in which case the calcium concentration is given
            for every neuron of the network.
        growth_rate : float, optional
            Initial growth rate of the calcium concentration. It must be
            positive. Defaults to `None`, in which case it is replaced with the
            inverse of the configuration's time step `config.delta`.
        decay_rate : float, optional
            Decay rate of the calcium concentration. It must be positive, and it
            should be much smaller than the initial growth rate. Defaults to
            `None`, in which case it is replaced with five percent of the
            initial growth rate.

        Returns
        -------
        array_like
            Calcium concentration with respect to time for every requested
            neuron, with neurons along the first axis and time along the second.
            If a single neuron was requested, it will be one-dimensional.

        Raises
        ------
        ValueError
            If `indices` is not a valid list of indices for neurons of the
            network.
        """
        if growth_rate is None:
            growth_rate = 1 / self.config.delta
        if decay_rate is None:
            decay_rate = .05 * growth_rate
        if isinstance(indices, int):
            return self._get_calcium_output(indices, growth_rate, decay_rate)
        valid_indices = np.arange(self.config.network.size())
        if indices is None:
            indices = valid_indices
        try:
            valid_indices[indices]
        except IndexError as error:
            raise ValueError(f'{indices} is not a valid list of indices for '
                             'neurons of the network.') from error
        calcium = np.zeros((N := len(indices), len(self.transition_times)))
        for j in range(N):
            calcium[j,:] = self._get_calcium_output(j, growth_rate, decay_rate)
        return calcium

    def close(self):
        """Delete all data attributes of the simulator."""
        super().close()
        del self._micro_states
        del self._transition_times

    def micro_output(self, fmt='ternary'):
        """Get the simulation's microscopic output.
        
        Get the microscopic state of the network with respect to time from the
        last simulation that was performed.

        Parameters
        ----------
        fmt : {'binary', 'ternary', 'calcium'}, optional
            Format of the neurons' states. If `'ternary'`, a neuron's state can
            take the values `1`, `1j` or `0`, associated with the *active*,
            *refractory* and *sensitive* states respectively. If `'binary'`, a
            neuron's state can take the values `1` or `0`, where `1` is still
            associated with the active state, but `0` is rather associated to
            any non-active state (sensitive and refractory). If `'calcium'`,
            the returned output is the default given by `calcium_output`. 
            Defaults to `'ternary'`.

        Returns
        -------
        array_like
            Microscopic state of the network with respect to time.

        Raises
        ------
        ValueError
            If `fmt` is passed an unexpected value.
        """
        self._check_if_run()
        if fmt == 'ternary':
            return self.micro_states
        if fmt == 'binary':
            return np.real(self.micro_states)
        if fmt == 'calcium':
            return self.calcium_output()
        raise ValueError(f'Unknown format {fmt} for microscopic states.')

    def reset(self):
        """Reset the simulator."""
        super().reset()
        self._states = None
        self._transition_times = [self.config.initial_time]
        self._micro_states = [self.config.micro_initial_state.copy()]
        self._reset_activation_rates()

    def single_run(self, do_step, rng, iterate):
        """Run a single simulation.

        Run a simulation to obtain a possible trajectory of the stochastic
        process which describes the evolution of the network. To obtain this
        trajectory, the Doob--Gillespie algorithme is used either with the
        direct method or with the first reaction method. See the
        [Notes](#simulator-single-run-notes) section below for more details
        about the algorithm.

        !!! note
            The recommended way to perform simulations of the stochastic process
            is *not* to use this method, but rather to use `SimpleSimulator.run`
            or `ChainSimulator.run`.

        Parameters
        ----------
        do_step : callable
            Dictates how to do the Monte Carlo step of the Doob--Gillespie
            algorithm. To be passed to `iterate`.
        rng : numpy.random.Generator
            Random number generator.
        iterate : callable
            Dictates how a complete iteration of the simulation is performed.
            This includes the Monte Carlo step as well as all other tasks that
            should be done at each time step.

        Notes {#simulator-single-run-notes}
        -----
        From the microscopic point of view, the evolution of the state of the
        whole network is described by a stochastic process. The simulation run
        by this method outputs a possible trajectory of this stochastic process,
        using the Doob--Gillespie algorithm, popularized by Gillespie in [3] and
        based on results of Doob [1,2]. The idea to pass from a state to another
        is first to find all of the states to which the network can go from the
        current one, with the corresponding transition rates. This information
        is in fact sufficient to determine the distribution of the time at which
        the next transition occur, and which one will occur. 

        In [3], Gillespie introduces two methods, called the *direct* and
        *first reaction* methods respectively, to choose the time interval until
        the next transition and the next state of the system.

         - **Direct method.** First, the total transition rate out of the
           current state is computed, and a time interval until the next
           transition is taken randomly knowing that it is exponentially
           distributed with parameter equal to this total out rate. Then a next
           state is chosen randomly knowing that the probability of going to a
           given other state is proportional to the corresponding transition
           rate.

         - **First reaction method.** For every possible next state, a time at
           which the corresponding transition could occur is randomly generated,
           knowing that this time is exponentially distributed with parameter
           equal to the transition rate. The transition that should occur first
           is chosen, and the state is updated accordingly.

        References
        ----------
         1. Doob, J. L. Topics in the Theory of Markoff Chains. *Transactions of
            the American Mathematical Society* **52**, 37--64 (1942).
            doi: [10.2307/1990152](https://doi.org/10.2307/1990152).
         2. Doob, J. L. Markoff Chains--Denumerable Case. *Transactions of the
            American Mathematical Society* **58**, 455--473 (1945).
            doi: [10.2307/1990339](https://doi.org/10.2307/1990339).
         3. Gillespie, D. T. A General Method for Numerically Simulating the
            Stochastic Time Evolution of Coupled Chemical Reactions. *Journal of
            Computational Physics* **22**, 403--434 (1976). doi:
            [10.1016/0021-9991(76)90041-3](
            https://doi.org/10.1016/0021-9991(76)90041-3).
        """
        t = self.transition_times[0]
        x = self.micro_states[0]
        while t < self.config.final_time:
            t, x = iterate(do_step, rng, t, x)
        self._micro_states = np.array(self.micro_states)
        self._transition_times = np.array(self.transition_times)
        self._update_states()

    def _check_sizes(self):
        """Check the consistency of the network's size and the initial state."""
        if len(self.config.micro_initial_state) != self.config.network.size():
            raise PopNetError('It seems that the size of the network has '
                              'changed since the microscopic initial state has '
                              'been set. It has to be reset. The network\'s '
                              'parameters might also have to be reset.')

    def _direct_method(self, rng, t, next_states, rates):
        """Obtain the next state and time from the direct method."""
        out_rate = np.sum(rates)
        threshold_rate = rng.random() * out_rate
        j = 0
        sum_of_rates = rates[0]
        while sum_of_rates < threshold_rate:
            sum_of_rates += rates[j+1]
            j += 1
        return j, (1 / out_rate) * np.log(1 / rng.random())

    def _first_reaction_method(self, rng, t, next_states, rates):
        """Obtain the next state and time from the first reaction method."""
        next_times = (1 / rates) * np.log(1 / rng.random(len(rates)))
        j = np.argmin(next_times)
        return j, next_times[j]

    def _get_calcium_output(self, j, growth_rate, decay_rate):
        """Get the calcium concentration in neuron *j* with respect to time."""
        binary_state = np.real(self.micro_states[:,j])
        all_activation_indices = np.nonzero(binary_state)[0]
        activation_indices = [k for i,k in enumerate(all_activation_indices)
                              if all_activation_indices[i-1] != k-1]
        calcium = np.zeros((len(activation_indices), 
                            len(self.transition_times)))
        for k, activation_index in enumerate(activation_indices):
            t = self.transition_times[activation_index:] 
            t0 = self.transition_times[activation_index]
            calcium[k,activation_index:] = ((1 - np.exp(-growth_rate * (t-t0)))
                                                * np.exp(-decay_rate * (t-t0)))
        calcium = np.sum(calcium, axis=0)
        return calcium

    def _iterate(self, do_step, rng, t, x):
        """Perform a single iteration of a simulation.

        Perform a single iteration of a simulation. 

        Parameters
        ----------
        do_step : function
            Dictates how the Monte Carlo step of Gillespie's algorithm.
        rng : Generator
            Random number generator.
        t : float
            Current time step.
        x : array_like
            Current state of the network.

        Returns
        -------
        float
            Next time step.
        array_like
            Next state of the network.
        """
        next_states, rates = self._next_states_and_rates(x)
        j, time_interval = do_step(rng, t, next_states, rates)
        x = next_states[j].copy()
        t += time_interval
        self.transition_times.append(t)
        self.micro_states.append(x.copy())
        return t, x

    def _make_activation_rate(self, j):
        """Generate the activation rate function for the `j`th neuron."""
        J = 0
        sum_sizes = self.config.network.populations[0].size
        while j > sum_sizes:
            sum_sizes += self.config.network.populations[J+1].size
            J += 1
        def act(x):
            b = np.dot(self.config.network.W[j], np.real(x)) + self.config.Q[J]
            if b < self.config.network.theta[j]:
                return 0.
            else:
                return self.config.network.alpha[j]
        return act

    def _next_states_and_rates(self, x):
        """Get all possible states to which the network can go from `x`.

        Knowing that the network is in state `x`, get all states which are 
        accessible next with the rates associated with each possible transition.

        Returns
        -------
        tuple of array_like
            The next possible states, and the associated transition rates. Both
            arrays are arranged so that the *j*th element of the transition rate
            vector is the rate at which the network can make a transition to the
            state corresponding to the *j*th row of the array of next states.
        """
        next_states = np.resize(x, (N := self.config.network.size(), N))
        rates = np.zeros(N)
        for j in range(N):
            next_states[j,j], rates[j] = self._next_state_and_rate(j, x)
        return next_states, rates

    def _next_state_and_rate(self, j, x):
        """Get the next state and transition rate of the `j`th neuron.

        Knowing that the network is in state `x`, get the next accessible state
        of `j`th neuron, with the rate at which this neuron will make a 
        transition. 

        Parameters
        ----------
        j : int
            The neuron for which to get the next state and transition rate.
        x : array_like
            The current state of the network.

        Returns
        -------
        tuple of complex and float
            The next state of the `j`th neuron with associated transition rate. 

        Raises
        ------
        ValueError
            If the `j`th neuron is in a non valid state. 
        """
        if (z := x[j]) == 0.:
            return 1., self.activation_rates[j](x)
        if z == 1.:
            return 1j, self.config.network.beta[j]
        if z == 1j:
            return 0., self.config.network.gamma[j]
        raise ValueError('The state of a neuron should always be 0, 1 or the'
                         'imaginary unit.')

    def _reset_activation_rates(self):
        """Defines the activation rate functions."""
        self._activation_rates = [self._make_activation_rate(j) 
                                  for j in range(self.config.network.size())]

    def _state_length(self):
        """Length of the macroscopic states."""
        return 2 * len(self.config.network.populations)

    def _update_states(self):
        """Update `states` based on `micro_states`.

        Compute the macroscopic states of the network from `micro_states`, and
        update `states` in consequence.
        """
        p = len(self.config.network.populations)
        states = np.zeros((len(self.transition_times), 2*p))
        j = 0
        for J, popJ in enumerate(self.config.network.populations):
            states[:,J]   = np.sum(np.real(self.micro_states[:,j:j+popJ.size]), 
                                axis=1) / popJ.size
            states[:,p+J] = np.sum(np.imag(self.micro_states[:,j:j+popJ.size]), 
                                axis=1) / popJ.size
            j += popJ.size
        self._states = states


class SimpleSimulator(Simulator):
    """Perform single simulations of the stochastic process on a network.

    The `SimpleSimulator` class extends the `Simulator` class to ease the task
    of running single simulations of the stochastic process. It has dedicated
    methods to run simulations and output a `Trajectory` instance. Its data
    attributes are the same as in the base class.

    """

    def run(self, method='direct', verbose=False):
        """Run a simulation.

        Run a simulation to obtain a possible trajectory of the stochastic
        process which describes the evolution of the network. To obtain this
        trajectory, we use the Doob--Gillespie algorithm, either with the direct
        or with the first reaction method. See `Simulator.single_run` for more
        details about the Doob--Gillespie algorithm.

        Parameters
        ----------
        method : {'direct', 'first reaction'}, optional
            Chooses which method is used to perform the Monte Carlo step in the
            Doob--Gillespie algorithm. Defaults to `'direct'`.
        verbose : bool, optional
            If `True`, the current time will be printed. Defaults to `False`.

        Raises
        ------
        PopNetError
            If the length of the microscopic initial state is different of the
            network's size.
        ValueError
            If an unexpected value is passed to `method`.
        """
        self._check_sizes()
        if method == 'direct':
            def do_step(rng,t,ns,r): return self._direct_method(rng,t,ns,r)
        elif method == 'first reaction':
            def do_step(rng,t,ns,r): return self._first_reaction_method(rng,t,ns,r)
        else:
            raise ValueError(f'Simulator does not know a method {method}.')
        if verbose:
            def iterate(do_step, rng, t, x):
                print(f't = {t:<25}', end='\r')
                return self._iterate(do_step, rng, t, x)
        else:
            def iterate(do_step, rng, t, x): 
                return self._iterate(do_step, rng, t, x)
        self.single_run(do_step, np.random.default_rng(), iterate)
        if verbose:
            print(30*' ', end='\r')
            print('Done!')
        self._success = True

    def save_output(self, name=None, folder=None):
        """Save the simulation's output to a text file.

        Extends the base class method by saving additionally the times at which
        transitions occur. This is done by saving the array
        `Simulator.transition_times` under *ID - name (times).txt*, where *ID*
        is the ID of the configuration used for the simulation, and *name* is
        `name`.

        Parameters
        ----------
        name : str, optional
            Name to give to the saved output. Defaults to `None`, in which case
            it is replaced with `'Trajectory'`.
        folder : str, optional
            If given, the file will be saved in this folder. It should already
            exist in the current directory. Defaults to `None`, in which case
            the file will be saved directly in the current directory.

        Returns
        -------
        name : str
            Name given to the saved output.

        Raises
        ------
        PopNetError
            If the simulation has not been performed yet.
        """
        name = super().save_output(name=name, folder=folder)
        filename = _format_filename(folder, self.config.ID, f'{name} (times)')
        np.savetxt(times_filename, self.transition_times, fmt='%+.12f')
        return name

    def _output_times(self):
        """Times array to output."""
        return self.transition_times

    def _output_type(self):
        """Type of output."""
        if len(self.config.network.populations) == 1:
            return TrajectoryOne
        return Trajectory


class ChainSimulator(Simulator):
    """Simulate multiple times the stochastic process on a network.

    The `ChainSimulator` class extends the `Simulator` class to ease the task
    of running many simulations of the stochastic process on the same network,
    with the same configuration, in order to obtain statistics. It has dedicated
    methods to run many simulations and output a `Statistics` instance. Its data
    attributes are the same as in the base, except for a new `samples`, which
    stores the trajectories obtained from simulations of the stochastic process.

    """

    @property
    def samples(self):
        """Samples of trajectories.

        Samples of trajectories of the stochastic process, once a simulation has
        been performed. It is a three dimensional array, where the first axis is
        time, the second is the macroscopic state component, and the third is
        associated to a given trajectory. It cannot be manually set nor deleted.
        """
        return self._samples

    def close(self):
        """Delete all data attributes of the simulator."""
        super().close()
        del self._samples

    def reset(self):
        """Reset the simulator."""
        super().reset()
        self._samples = None

    def run(self, method='direct', verbose=False):
        """Run multiple simulations.

        Run multiple simulations of the stochastic process which describes the
        evolution of the network, in order to obtain a sample of possible
        trajectories. Each single simulation is perforfmed with the
        Doob--Gillespie algorithm, either with the direct or the first reaction
        method. See `Simulator.single_run` for more details about the
        Doob--Gillespie algorithm.

        Parameters
        ----------
        method : {'direct', 'first reaction'}, optional
            Chooses which method is used to perform the Monte Carlo step in the
            Doob--Gillespie algorithm. Defaults to `'direct'`.
        verbose : bool, optional
            If `True`, a progression bar will be printed to show how much of the
            `n` simulations have been performed. Defaults to `False`.

        Raises
        ------
        PopNetError
            If the length of the microscopic initial state is different of the
            network's size.
        ValueError
            If an unexpected value is passed to `method`.
        """
        self._check_sizes()
        samples = np.zeros((1+self.config.iterations, self._state_length(), 
                            self.config.executions))
        if method == 'direct':
            def do_step(rng,t,ns,r): return self._direct_method(rng,t,ns,r)
        elif method == 'first reaction':
            def do_step(rng,t,ns,r): return self._first_reaction_method(rng,t,ns,r)
        else:
            raise ValueError(f'Simulator does not know a method {method}.')
        if verbose:
            def progress(rg): return tqdm(rg)
        else:
            def progress(rg): return rg
        rng = np.random.default_rng()
        for j in progress(range(self.config.executions)):
            self.single_run(do_step, rng, self._iterate)
            for J in range(self._state_length()):
                samples[:,J,j] = np.interp(self.times, self.transition_times, 
                                           self.states.T[J])
            self.reset()
        self._samples = samples
        self._success = True

    def save_output(self, name=None, folder=None):
        """Save the samples obtained from simulations.

        Overrides the base class method to save the samples of trajectories
        obtainedfrom numerical simulations rather than the macroscopic states
        they yield. Each state component *X* is saved in its own file, named
        *ID - name X.txt*, where *ID* is the ID of the configuration used for
        the simulations, and *name* is `name`.

        Parameters
        ----------
        name : str, optional
            Name to give to the saved samples. Defaults to `None`, in which case
            it is replaced with `'Sample'`.
        folder : str, optional
            If given, the files will be saved in this folder. It should already
            exist in the current directory. Defaults to `None`, in which case
            the files will be saved directly in the current directory.

        Returns
        -------
        name : str
            Name given to the saved samples.

        Raises
        ------
        PopNetError
            If the simulations have not been performed yet.
        """
        self._check_if_run()
        name = self._output_type()._get_sample_name(name)
        for J, X in enumerate(self.config._variables[:self._state_length()]):
            filename = _format_filename(folder, self.config.ID, f'{name} {X}')
            h = (f'In each column are the values of {X} with respect to time '
                 'for a given trajectory.')
            np.savetxt(filename, self.samples[:,J,:], fmt='%+.12f', header=h)
        return name

    def _output_states(self):
        """States array to output."""
        return self.samples

    def _output_type(self):
        """Type of output."""
        if len(self.config.network.populations) == 1:
            return StatisticsOne
        return Statistics


class Result:
    """Results generated using PopNet functions.

    The purpose of the `Result` class is to handle easily the outputs of
    numerical simulations performed by PopNet functions. The base class `Result`
    has several methods, listed in the [Methods](#result-methods) section below,
    to easily initialize and setup a [`matplotlib`](https://matplotlib.org/)
    figure with predefined formatting, allowing to easily produce many figures
    in a consistent format. 

    Although limited features would be available with the `Result` class alone,
    it is not inteded to be used by itself, but rather through its subclasses:
    `Solution`, `MeanField`, `Trajectory`, `Statistics` and `Spectrum`. Each one
    of these subclasses implements other features specific to a given result
    case.

    Parameters
    ----------
    config : Configuration
        The configuration used to obtain the result.
    states : array_like
        The state of the network with respect to time.
    times : array_like
        An array representing time.
    name : str, optional
        A name associated to the result. Defaults to `None`, in which case it is
        replaced with the name of the class.

    Attributes
    ----------
    config : Configuration
        The configuration used to obtain the result. See `Result.config`.
    name : str
        Name associated to the result. See `Result.name`.
    times : array_like
        Time.
    fig : matplotlib.figure.Figure
        A `matplotlib` figure. See `Result.fig`.
    ax : matplotlib.axes.Axes
        The axes of `fig`. See `Result.ax`.
    colors : dict
        Colors for each state variable associated with the result. See
        `Result.colors`.
    plot : dict
        Plotting methods for each state variable associated with the result. See
        `Result.plot`.
    A, R, S : array_like
        Vectors of state variables with respect to time.
    CAA, CRR, CSS, CAR, CAS, CRS : array_like
        Matrices of covariance between state variables with respect to time, or
        `None` if no such covariances are defined for a `Result` subclass.

    Methods {#result-methods}
    -------
     - `Result.initialize_graph` :
        Initialize the `matplotlib` figure.
     - `Result.make_legend` :
        Make a legend for the figure.
     - `Result.setup_graph` :
        Setup the axes of the figure.
     - `Result.end_graph` :
        End the figure and show it or save it.

    Raises
    ------
    TypeError
        If `config` is not a `Configuration` instance.

    """

    def __init__(self, config, states, times, name=None):
        if not isinstance(config, Configuration):
            raise TypeError('The configuration used with a Result instance '
                            'should be a Configuration instance.')
        self._config = config
        self.name = self._get_name(name)

        self._init_abscissa(times)
        self._init_states_dict(states)
        self._init_colors()
        self._init_plot_methods()

        self.fig = None
        self.ax = None

    @classmethod
    def load(cls, ID, name=None, config=None, times=None, folder=None):
        """Load the result associated with the ID.

        Load the result obtained when using the configuration of ID `ID`. The
        array representing the state of the network with respect to time is
        expected to be in a file named *ID - name.txt*, where *ID* and *name*
        are indeed `ID` and `name`. 

        Parameters
        ----------
        ID : str
            The ID of the configuration used to obtain this result. 
        name : str, optional
            The name associated with the result. Defaults to `None`, in which
            case it is replaced with the name of the class.
        config : Configuration, optional
            Configuration to associate with the result. If given, it should have
            the ID `ID`. Defaults to `None`, in which case it is loaded. 
        times : array_like, optional
            Times array to associate to the result. Defaults to `None`, in
            which case it is computed from the configuration. This will not work
            for all `Result` subclasses.
        folder : str, optional
            Folder in which the file is located, which should be placed in the
            current directory. Defaults to `None`, in which case the file is
            assumed to be located directly in the current directory.

        Returns
        -------
        Result
            The loaded result. 

        Raises
        ------
        TypeError
            If `config` is neither `None` nor a `Configuration` instance.
        PopNetError
            If `config` has a different ID than `ID`. 
        FileNotFoundError
            If no file is found with the expected name.
        """
        config = self._check_config(config, ID)
        if times is None:
            times = np.linspace(config.initial_time, config.final_time, 
                                1 + config.iterations)
        name = self._get_name(name)
        filename = _format_filename(folder, ID, name)
        try:
            states = np.loadtxt(filename, dtype=float)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'It seems that the result has not been obtained yet from the '
                f'configuration {ID}.') from error
        return cls(config, states, times, name)

    @property
    def config(self):
        """Configuration used to obtain this result.

        Configuration that was used to obtain this result. It is set at
        initialization, and cannot be set or deleted afterwards.
        """
        return self._config

    @property
    def name(self):
        """Name of the result. It has to be a string."""
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError('The name associated with a Result should be given '
                            'as a string.')
        self._name = new_name

    @property
    def times(self):
        """Time.

        An array representing time. It is the independant variable with respect
        to which state components are given. It cannot be set nor deleted.
        """
        return self._times

    @property
    def fig(self):
        """A figure used to draw graphical representations of the result.

        A [`matplotlib.figure.Figure`](
        https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.figure.Figure.html)
        object that can be used to represent the result in various ways. This is
        the figure where `Result`'s methods can plot curves. It is set
        automatically when `Result.initialize_graph` is called. It cannot be
        deleted manually.
        """
        return self._fig

    @fig.setter
    def fig(self, new_value):
        if new_value is None:
            pass
        elif not isinstance(new_value, mpl.figure.Figure):
            raise TypeError('Result\' fig attribute should be a matplotlib.'
                            'figure.Figure object.')
        self._fig = new_value

    @property
    def ax(self):
        """Axes of the current figure.

        A [`matplotlib.axes.Axes`](
        https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
        object correponding to the axes of `Result.fig`. It is set
        automatically when `Result.initialize_graph` is called. It cannot be
        deleted manually.
        """
        return self._ax

    @ax.setter
    def ax(self, new_value):
        if new_value is None:
            pass
        elif not isinstance(new_value, mpl.axes.Axes):
            raise TypeError('Result\' ax attribute should be a matplotlib.axes'
                            '.Axes object.')
        self._ax = new_value

    @property
    def colors(self):
        """Colors associated with the solution's components.

        Colors associated with the result's components, to be used in graphics.
        It should be a dictionary whose keys are strings representing possible
        components, and whose values are lists (or lists of lists) of valid
        `matplotlib` colors associated to each population (or pair of
        populations). For example, `solution.colors['A'][J]` is the color
        associated to the activity of the *J*th population of the network for
        the result `solution`. By default, it is a `PopNetDict`, so in the case
        of a single population, a new color can be given as is, and it will
        automatically be put in a list. It cannot be deleted.
        """
        return self._colors

    @colors.setter
    def colors(self, new_colors):
        if not isinstance(new_colors, dict):
            raise TypeError('The colors passed to a Result instance must be '
                            'stored in a dictionary.')
        self._colors = new_colors

    @property
    def plot(self):
        """Dictionary of methods to plot state variables.

        If `X` is a string representing a state variable (that is, either `'A'`,
        `'R'` or `'S'`), then `plot[X]` is a list whose `J`th element is a
        method which plots the state variable `X` of the `J`th population.
        Similary, if covariances are defined for this result, then
        `plot[CXY][J][K]` is a method which plots the covariance between `X` and
        `Y` for the `J`th and `K`th population respectively. A similar pattern
        works for third central moments if applicable, with keys of the form
        `XYZ` where `X`, `Y` and `Z` are all state variables.

        All methods to plot accept keyword arguments that can be passed to the
        `plot` method of a [`matplotlib.axes.Axes`](
        https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
        object. This attribute cannot be set nor deleted.
        """
        return self._plot

    @property
    def A(self):
        """Get the state components associated with *A*.

        Get the list of components associated with the state variable *A*, that
        is, the active fraction of a population. It cannot be set nor deleted.
        """
        return self._states_dict['A']

    @property
    def R(self):
        """Get the state components associated with *R*.

        Get the list of components associated with the state variable *R*, that
        is, the refractory fraction of a population. It cannot be set nor
        deleted.
        """
        return self._states_dict['R']

    @property
    def S(self):
        """Get the state components associated with *S*.

        Get the list of components associated with the state variable *S*, that
        is, the sensitive fraction of a population. It cannot be set nor
        deleted.
        """
        return self._states_dict['S']

    @property
    def CAA(self):
        """Covariances between active fractions of populations.

        Matrix of covariances between active fractions of populations with
        respect to time, or `None` if no such matrix is defined. It cannot be
        set nor deleted. 
        """
        try:
            return self._states_dict['CAA']
        except KeyError:
            pass

    @property
    def CRR(self):
        """Covariances between refractory fractions of populations.

        Matrix of covariances between refractory fractions of populations with
        respect to time, or `None` if no such matrix is defined. It cannot be
        set nor deleted. 
        """
        try:
            return self._states_dict['CRR']
        except KeyError:
            pass

    @property
    def CSS(self):
        """Covariances between sensitive fractions of populations.

        Matrix of covariances between sensitive fractions of populations with
        respect to time, or `None` if no such matrix is defined. It cannot be
        set nor deleted. 
        """
        try:
            return self._states_dict['CSS']
        except KeyError:
            pass

    @property
    def CAR(self):
        """Covariances between active and refractory fractions of populations.

        Matrix of covariances between active and refractory fractions of
        populations with respect to time, or `None` if no such matrix is
        defined. It cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CAR']
        except KeyError:
            pass

    @property
    def CAS(self):
        """Covariances between active and sensitive fractions of populations.

        Matrix of covariances between active and sensitive fractions of
        populations with respect to time, or `None` if no such matrix is
        defined. It cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CAS']
        except KeyError:
            pass

    @property
    def CRS(self):
        """Covariances between refractory and sensitive fractions of populations.

        Matrix of covariances between refractory and sensitive fractions of
        populations with respect to time, or `None` if no such matrix is
        defined. It cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CRS']
        except KeyError:
            pass

    def default_graph(self, ncol=3, show=True, savefig=False, **kwargs):
        """Make a graphic with default parameters.

        Make a graphic with default parameters. If the figure is saved, it will
        be under *ID - name.png*, where *ID* and *name* are the corresponding
        attributes of the result.

        Parameters
        ----------
        ncol : int, optional
            Number of columns of the legend. Defaults to 3.
        show : bool
            Decides if the figure is shown or not. Defaults to `True`.
        savefig : bool, optional
            Decides if the figure is saved or not. Defaults to `False`.
        **kwargs
            Keyword arguments to be passed to an internal method that decides
            what is plotted on the figure. Valid keyword arguments are given in
            the [Other Parameters](#result-default-graph-other-parameters)
            section below.
        
        Other Parameters {#result-default-graph-other-parameters}
        ----------------
        expectation : bool, optional
            Decides whether expectations are plotted on the graph. Valid in
            `Solution`, `Statistics` and `Spectrum`. Defaults to `True`.
        variances : bool, optional
            Decides whether variances are plotted on the graph. Valid in
            `Solution`, `Statistics` and `Spectrum`. Defaults to `True`.
        covariances : bool, optional
            Decides whether non-symmetric covariances are plotted on the graph.
            Valid in `Solution`, `Statistics` and `Spectrum`. Defaults to
            `False`.
        third_moments : bool, optional
            Decides whether third central moments are plotted on the graph.
            Valid in `Statistics` and `Spectrum`. Defaults to `False`.
        """
        self.initialize_graph()
        self._default_plots(**kwargs)
        self.setup_graph()
        self.make_legend(ncol=ncol)
        self.end_graph(show=show, savefig=savefig)

    def end_graph(self, name=None, show=True, savefig=False, folder=None):
        """End a graphic.

        End a graphic started with `Result.initialize_graph`. If the figure is
        saved, it will be under *ID - name.png*, where *ID* and *name* are the
        corresponding attributes of the result.

        Parameters
        ----------
        name : str, optional
            Name to give to the figure if saved. Defaults to `None`, in which
            case the `name` attribute is used.
        show : bool, optional
            Decides if the figure is shown or not. Defaults to `True`.
        savefig : bool, optional
            Decides if the figure is saved or not. Defaults to `False`.
        folder : str, optional
            A folder in which the figure can be saved. If given, it should
            already exist in the current directory. Defaults to `None`, in which
            case the figure is saved in the current directory.

        Raises
        ------
        PopNetError
            If the figure has not been initialized yet.
        """
        self._check_if_initialized()
        if savefig:
            if name is None:
                name = self.name
            filename = _format_filename(folder, self.config.ID, name, 'png')
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close(self.fig)

    def get_spectrum(self, name=None):
        """Get the spectrum of this result.

        Get a `Spectrum` instance corresponding to the present instance where
        each state component is replaced by its real fast Fourier transform.

        Parameters
        ----------
        name : str, optional
            Name to associate to the spectrum. Defaults to `None`, in which case
            it is replaced with `Spectrum`.

        Returns
        -------
        Spectrum
            The spectrum of the result.
        """
        return Spectrum(self.config, self._states_dict, self.times, 
                        self._default_name(), name)

    def initialize_graph(self, usetex=False, figsize=(5,3.75), dpi=150, **kwargs):
        """Initialize a graphic.
        
        Create a `matplotlib` figure to plot results, and set the `fig` and `ax`
        attributes to refer to the [`matplotlib.figure.Figure`](
        https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.figure.Figure.html)
        and [`matplotlib.axes.Axes`](
        https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
        objects corresponding to this figure.

        Parameters
        ----------
        usetex : bool, optional
            Determines if TeX is used in the figure. If it is used, the LaTeX
            packages `newpxtext` and `newpxmath` are also loaded to use the
            font Palatino. Defaults to `False`.
        figsize : tuple of float, optional
            Width and height of the figure in inches. Defaults to `(5, 3.75)`.
        dpi : int
            Resolution of the figure in dots per inches. Defaults to 150.
        **kwargs
            Keywords arguments to be passed to [`matplotlib.pyplot.figure`](
            https://tinyurl.com/plt-figure). 
        """
        self.fig, self.ax = init_standard_graph(
            colors=False, usetex=usetex, figsize=figsize, dpi=dpi, **kwargs)

    def make_legend(self, fontsize=10, ncol=3, handletextpad=0.5, **kwargs):
        """Generate a legend for the figure.

        Generate a legend for the figure with default options.

        Parameters
        ----------
        fontsize : int, optional
            Fontsize of the legend's labels. Defaults to 10.
        ncol : int, optional
            Number of columns of the legend. Defaults to 3.
        handletextpad : float, optional
            Padding of the labels. Defaults to 0.5.
        **kwargs
            Keyword arguments to be passed to the `legend` method of the `ax`
            attribute. Recall that it is a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        leg = self.ax.legend(fontsize=fontsize, ncol=ncol, 
                             handletextpad=handletextpad, **kwargs)
        for lego in leg.legendHandles:
            lego.set_linewidth(2)
        return leg

    def setup_graph(self):
        """Setup a graphic.

        Setup a graphic started with `Result.initialize_graph`.

        Raises
        ------
        PopNetError
            If the figure has not been initialized yet.
        """
        self._check_if_initialized()

    def _add_colors_items_one(self):
        """Add items to the `colors` dictionary. 

        Add to the `colors` dictionary items corresponding to state variables
        involving *one* population, and define some default colors.
        """
        for X in ['A', 'R', 'S']:
            self.colors[X] = [None for pop in self.config.network.populations]
        if (p := len(self.config.network.populations)) == 1:
            self.colors['A'][0] = (150/255,10/255,47/255)
            self.colors['R'][0] = 'midnightblue'
            self.colors['S'][0] = 'goldenrod'
        elif p == 2:
            self.colors['A'] = ['midnightblue', (150/255,10/255,47/255)]
            self.colors['R'] = ['royalblue', 'crimson']
            self.colors['S'] = ['skyblue', (243/255,125/255,148/255)]

    def _add_colors_items_two(self):
        """Add items to the `colors` dictionary. 

        Add to the `colors` dictionary items corresponding to variables
        involving *two* populations, and define some default colors.
        """
        for XY in ['AA', 'RR', 'SS', 'AR', 'AS', 'RS']:
            key = f'C{XY}'
            self.colors[key] = [[None for p1 in self.config.network.populations]
                                      for p2 in self.config.network.populations]
        if (p := len(self.config.network.populations)) == 1:
            self.colors['CAA'][0][0] = 'salmon'
            self.colors['CRR'][0][0] = 'skyblue'
            self.colors['CSS'][0][0] = 'gold'
            self.colors['CAR'][0][0] = 'violet'
            self.colors['CAS'][0][0] = (255/255,180/255,0)
            self.colors['CRS'][0][0] = 'springgreen'
        elif p == 2:
            self.colors['CAA'] = [['seagreen', None], [None, 'purple']]
            self.colors['CRR'] = [['mediumseagreen', None], 
                                  [None, 'mediumorchid']]
            self.colors['CSS'] = [['springgreen', None], [None, 'violet']]

    def _add_colors_items_three(self):
        """Add items to the `colors` dictionary. 

        Add to the `colors` dictionary items corresponding to variables
        involving *three* populations.
        """
        keys = ['AAA', 'AAR', 'AAS', 'ARR', 'ARS', 
                'ASS', 'RRR', 'RRS', 'RSS', 'SSS']
        for k in keys:
            self.colors[k] = [[[None for p1 in self.config.network.populations]
                                     for p2 in self.config.network.populations]
                                     for p3 in self.config.network.populations]

    @staticmethod
    def _check_config(config, ID):
        """Check if `config` is a valid configuration when loading a result."""
        if config is None:
            return load_config(ID)
        if not isinstance(config, Configuration):
            raise TypeError('The configuration to associate with a loaded '
                            'result should be a Configuration instance.')
        if config.ID != ID:
            raise PopNetError('The configuration to associate to a result '
                              f'loaded from ID {ID} should have the same ID.')
        return config

    def _check_if_initialized(self):
        """Check if the figure is already initialized."""
        if self.fig is None:
            raise PopNetError('The graph must be initialized before to be '
                              'ended. Call initialize_graph() first.')

    @classmethod
    def _default_name(cls):
        """Default name given to instances."""
        return cls.__name__

    def _default_plots(self, one=True, symmetric=False, nonsymmetric=False,
                       three=False):
        """Add plots on the default graph."""
        if one:
            self._plot_all_one()
        self._plot_all_two(symmetric=symmetric, nonsymmetric=nonsymmetric)
        if three:
            self._plot_all_three()

    @classmethod
    def _get_name(cls, name):
        """Return `name`, or the default name if `name` is `None`."""
        if name is None:
            return cls._default_name()
        else:
            return name

    def _init_abscissa(self, times):
        """Initialize array related to the independant variable."""
        self._times = times

    def _init_colors(self):
        """Initialize the colors associated to state variables."""
        self.colors = PopNetDict()
        self._add_colors_items_one()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        self._plot = self._plot_dict_one()

    def _init_states_dict(self, states):
        """Initialize the state variables dictionary."""
        transposed_states = np.transpose(states)
        A = transposed_states[: (p := len(self.config.network.populations))]
        R = transposed_states[p : 2*p]
        S = 1 - A - R
        self._states_dict = {'A': A, 'R': R, 'S': S}

    def _label_one(self, X, J):
        pass

    def _label_two(self, CXY, J, K):
        pass

    def _label_three(self, XYZ, J, K, L):
        pass

    def _make_plot_one(self, X, J, states=None, label_func=None, lw=None, ls=None):
        """Define a plotting method for a given state variable.

        Define a method to plot the state variable `X` for the population `J`,
        labeling the curve with `label_func` and taking the data in `states`.
        """
        if states is None:
            states = self._states_dict
        if label_func is None:
            label_func = self._label_one
        def f(add_label=True, lw=lw, ls=ls, **kwargs):
            self._check_if_initialized()
            if add_label:
                label = label_func(X, J)
            else:
                label = None
            line, = self.ax.plot(self.times, states[X][J], label=label, lw=lw,
                                 ls=ls, color=self.colors[X][J], **kwargs)
            return line
        return f

    def _make_plot_two(self, CXY, J, K):
        """Define a plotting method for a covariance.

        Define a method to plot the covariance `CXY` for the `J`th and `K`th
        population.
        """
        def f(**kwargs):
            self._check_if_initialized()
            label = self._label_two(CXY[1:], J, K)
            covariance = self._states_dict[CXY][J][K]
            line, = self.ax.plot(self.times, covariance, label=label,
                                 color=self.colors[CXY][J][K], **kwargs)
            return line
        return f

    def _make_plot_three(self, XYZ, J, K, L):
        """Define a plotting method for a third central moment.
        
        Define a method to plot the third central moment for the state variables
        `X`, `Y` and `Z` for the `J`th, `K`th and `L`th population respectively.
        """
        def f(verbose=True, **kwargs):
            self._check_if_initialized()
            try:
                moment = self._states_dict[XYZ][J][K][L]
            except (KeyError, IndexError):
                moment = None
            if moment is None:
                if verbose:
                    warn('The third central moment requested to plot is not '
                         'available. If this result has third central moments, '
                         'there should be a transposition of indices that '
                         'allows to get the same moment in another way.',
                         category=PopNetWarning, stacklevel=2)
                return None
            label = self._label_three(XYZ, J, K, L)
            line, = self.ax.plot(self.times, moment, label=label, 
                                 color=self.colors[XYZ][J][K][L], **kwargs)
            return line
        return f

    def _plot_all_one(self, **kwargs):
        """Plot all state variables associated with *one* population."""
        for J in range(len(self.config.network.populations)):
            for X in ['A', 'R', 'S']:
                self.plot[X][J](**kwargs)

    def _plot_all_two(self, symmetric=True, nonsymmetric=True, **kwargs):
        """Plot all state variables associated with *two* populations.

        Parameters
        ----------
        symmetric : bool, optional
            Decides whether variances are plotted. Defaults to `True`.
        nonsymmetric : bool, optional
            Decides whether non-symmetric covariances are plotted. Defaults to
            `True`.
        """
        p = len(self.config.network.populations)
        for J, K in np.ndindex((p,p)):
            for CXX in ['CAA', 'CRR', 'CSS']:
                if J == K and symmetric:
                    self.plot[CXX][J][K](**kwargs)
                if J < K and nonsymmetric:
                    self.plot[CXX][J][K](**kwargs)
            for CXY in ['CAR', 'CAS', 'CRS']:
                if nonsymmetric:
                    self.plot[CXY][J][K](**kwargs)

    def _plot_all_three(self, **kwargs):
        """Plot all state variables associated with *three* populations."""
        p = len(self.config.network.populations)
        for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARR',
                    'ASS', 'RRR', 'RRS', 'RSS', 'SSS']:
            for J, K, L in np.ndindex((p,p,p)):
                self.plot[XYZ][J][K][L](verbose=False, **kwargs)

    def _plot_dict_one(self, **kwargs):
        """Return a dictionary of plotting methods for *one* population."""
        p = len(self.config.network.populations)
        return {X: [self._make_plot_one(X, J, **kwargs) for J in range(p)]
                for X in ['A', 'R', 'S']}

    def _plot_dict_two(self):
        """Return a dictionary of plotting methods for *two* populations."""
        p = len(self.config.network.populations)
        return {CXY: [[self._make_plot_two(CXY, J, K)
                       for K in range(p)] for J in range(p)]
                for CXY in ['CAA', 'CRR', 'CSS', 'CAR', 'CAS', 'CRS']}

    def _plot_dict_three(self):
        """Return a dictionary of plotting methods for *three* populations."""
        p = len(self.config.network.populations)
        return {XYZ: [[[self._make_plot_three(XYZ, J, K, L) for L in range(p)]
                        for K in range(p)] for J in range(p)]
                for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARS',
                            'ASS', 'RRR', 'RRS', 'RSS', 'SSS']}

    def _set_xlim(self, xlim):
        """Set the limits of the horizontal axis of a figure."""
        if xlim == 'time':
            self.ax.set_xlim([self.config.initial_time, self.config.final_time])
        elif xlim == 'freqs':
            self.ax.set_xlim([0, self.freqs[-1]])
        elif xlim == 'config':
            upper = (1+self.config.iterations) / self.config.final_time
            self.ax.set_xlim([0, upper])
        elif xlim == 'unbounded':
            pass

    def _set_ylim(self, ylim):
        """Set the limits of the vertical axis of a figure."""
        if ylim == 'expectations':
            self.ax.set_ylim([0, 1])
        elif ylim == 'covariances':
            self.ax.set_ylim([-1/4, 1])
        elif ylim == 'unbounded':
            pass


class ResultOne(Result):
    """Adapts the `Result` class to the special case of a single population.

    The `ResultOne` class adapts the `Result` class for the case where the
    network associated with the configuration used has only one population.

    The main changes are that the attributes of the form `X` or `CXY` are
    overridden to return directly the corresponding quantity with respect to
    time instead of returning them as lists containing a single element. In the
    same way, the `plot` dictionary values are overridden to be methods rather
    than lists of methods. Also, the default name for a `ResultOne` class
    trims out the `'One'` suffix of the class' name to leave only `Result`.

    Other changes are implementation details, except in the methods where
    changes are explicitely noted.

    Raises
    ------
    PopNetError
        If the network associated to the configuratio ndoes not have precisely
        one population.

    """

    def __init__(self, config, states, times, name=None):
        super().__init__(config, states, times, name)
        if (p := len(self.config.network.populations)) != 1:
            raise PopNetError('The ResultOne class should only be used for '
                              'cases were the network has indeed a single '
                              f'population. Here the network has {p}.')

    @property
    def A(self):
        """Get the state component associated with *A*.

        Get the state component associated with the state variable *A*, that
        is, the active fraction of the network. It cannot be set nor deleted.
        """
        return self._states_dict['A'][0]

    @property
    def R(self):
        """Get the state component associated with *R*.

        Get the list of component associated with the state variable *R*, that
        is, the refractory fraction of the network. It cannot be set nor
        deleted.
        """
        return self._states_dict['R'][0]

    @property
    def S(self):
        """Get the state component associated with *S*.

        Get the list of component associated with the state variable *S*, that
        is, the sensitive fraction of the network. It cannot be set nor deleted.
        """
        return self._states_dict['S'][0]

    @property
    def CAA(self):
        """Variance of the active fraction of the network.

        Variance of the active fraction of the network with respect to time, or
        `None` if no such variance is defined. It cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CAA'][0][0]
        except KeyError:
            pass

    @property
    def CRR(self):
        """Variance of the refractory fraction of network.

        Variance of the refractory fractions of the network with respect to
        time, or `None` if no such variance is defined. It cannot be set nor
        deleted. 
        """
        try:
            return self._states_dict['CRR'][0][0]
        except KeyError:
            pass

    @property
    def CSS(self):
        """Variance of the sensitive fraction of network.

        Variance of the sensitive fractions of the network with respect to
        time, or `None` if no such variance is defined. It cannot be set nor
        deleted. 
        """
        try:
            return self._states_dict['CSS'][0][0]
        except KeyError:
            pass

    @property
    def CAR(self):
        """Covariance between active and refractory fractions of the network.

        Covariance between the active and refractory fractions of the network
        with respect to time, or `None` if no such covariance is defined. It
        cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CAR'][0][0]
        except KeyError:
            pass

    @property
    def CAS(self):
        """Covariance between active and sensitive fractions of the network.

        Covariance between the active and sensitive fractions of the network
        with respect to time, or `None` if no such covariance is defined. It
        cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CAS'][0][0]
        except KeyError:
            pass

    @property
    def CRS(self):
        """Covariance between refractory and sensitive fractions of the network.

        Covariance between the efractory and sensitive fractions of the network
        with respect to time, or `None` if no such covariance is defined. It
        cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CRS'][0][0]
        except KeyError:
            pass

    def get_spectrum(self, name=None):
        """Get the spectrum of this result.
        
        Overrides the base class method to return a `SpectrumOne` instance
        instead of a `Spectrum` instance.

        See Also
        --------
        Result.get_spectrum
        """
        return SpectrumOne(self.config, self._states_dict, self.times, 
                           self._default_name(), name)

    @classmethod
    def _default_name(cls):
        """Default name given to instances."""
        return cls.__name__.replace('One', '')

    def _plot_all_one(self, **kwargs):
        """Plot all state variables associated with *one* population."""
        for X in ['A', 'R', 'S']:
            self.plot[X](**kwargs)

    def _plot_all_two(self, symmetric=True, nonsymmetric=True, **kwargs):
        """Plot all state variables associated with *two* populations."""
        if symmetric:
            for CXX in ['CAA', 'CRR', 'CSS']:
                self.plot[CXX](**kwargs)
        if nonsymmetric:
            for CXY in ['CAR', 'CAS', 'CRS']:
                self.plot[CXY](**kwargs)

    def _plot_all_three(self, **kwargs):
        """Plot all state variables associated with *three* populations."""
        p = len(self.config.network.populations)
        for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARR',
                    'ASS', 'RRR', 'RRS', 'RSS', 'SSS']:
            self.plot[XYZ](verbose=False, **kwargs)

    def _plot_dict_one(self, **kwargs):
        """Return a dictionary of plotting methods for *one* population."""
        return {X: self._make_plot_one(X, 0, **kwargs) for X in ['A', 'R', 'S']}

    def _plot_dict_two(self):
        """Return a dictionary of plotting methods for *two* populations."""
        return {CXY: self._make_plot_two(CXY, 0, 0)
                for CXY in ['CAA', 'CRR', 'CSS', 'CAR', 'CAS', 'CRS']}

    def _plot_dict_three(self):
        """Return a dictionary of plotting methods for *three* populations."""
        return {XYZ: self._make_plot_three(XYZ, 0, 0, 0)
                for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARS',
                            'ASS', 'RRR', 'RRS', 'RSS', 'SSS']}


class MeanField(Result):
    """Represent solutions of the mean field dynamical system.

    The `MeanField` class extends the `Result` class for the case where the
    result is a solution obtained from a numerical integration of the dynamical
    system from the Wilson--Cowan model with refractory state. Specifically, it
    adds methods to plot state components and it extends the options to setup a
    figure. Other changes are implementation details.

    """

    def plot_expectations(self, **kwargs):
        """Plot all expectations of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_one(**kwargs)

    def setup_graph(self, time_units='ms'):
        """Setup a graphic.

        Setup a graphic started with `Result.initialize_graph`. Extends the base
        class method by setting limits to the figure's axes. 

        Parameters
        ----------
        time_units : str, optional
            Time units to be displayed on the figure. Defaults to `'ms'`, which
            assumes that the transition rates are measured in kHz.
        """
        super().setup_graph()
        self.ax.set_xlabel(f'Time [{time_units}]', fontsize=10)
        self._set_xlim('time')
        self._set_ylim('expectations')

    def _label_one(self, X, J):
        """Label for the expectation of `X` for the `J`th population."""
        return f'$\\mathcal{{{X}}}_{{{self.config.network.populations[J].ID}}}$'


class MeanFieldOne(ResultOne, MeanField):
    """Represent solutions of the mean field dynamical system for one population.

    The `MeanFieldOne` class combines the `ResultOne` and `MeanField`
    classes to represent solutions obtained from numerical integrations of the
    dynamical system from the Wilson--Cowna model, in the case where the network
    has a single population. Its only extensions from the base classes are
    implementation details.

    """

    def _label_one(self, X, J):
        """Label for the expectation of `X`."""
        return f'$\\mathcal{{{X}}}$'


class Solution(Result):
    """Represent solutions of the reduced dynamical system.

    The `Solution` class extends the `Result` class for the case where the
    result is a solution obtained from a numerical integration of the dynamical
    system from the extended Wilson--Cowan model. Specifically, it adds new
    methods to plot state components, and it extends the options to setup a
    figure. Other changes are implemententation details.

    """

    def plot_expectations(self, **kwargs):
        """Plot all expectations of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_one(**kwargs)

    def plot_variances(self, **kwargs):
        """Plot all variances of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_two(symmetric=True, nonsymmetric=False, **kwargs)

    def plot_covariances(self, **kwargs):
        """Plot all non-symmetric covariances of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_two(symmetric=False, nonsymmetric=True, **kwargs)

    def setup_graph(self, time_units='ms', xlim='time', ylim='expectations'):
        """Setup a graphic.

        Setup a graphic started with `Result.initialize_graph`. Extends the base
        class method by labeling the horizontal axis, and by setting limits to
        the figure's axes.

        Parameters
        ----------
        time_units : str, optional
            Time units to be displayed on the figure. Defaults to `'ms'`, which
            assumes that the transition rates are measured in kHz.
        xlim : {'time', 'unbounded'}, optional
            Decides how the horizontal axis is bounded. If `'time'`, it will be
            bounded by the initial and final times of the configuration. If
            `'unbounded'`, it will not be bounded. Defaults to `'time'`.
        ylim : {'expectations', 'covariances', 'unbounded'}, optional
            Decides how the vertical axis is bounded. If `'expectations'`, it
            will be bounded between 0 and 1. If `'covariances'`, it will be
            bounded between -1/4 and 1. If `'unbounded'`, it will not be
            bounded. Defaults to `'expectations'`.

        Raises
        ------
        ValueError
            If `xlim` or `ylim` is given a non-valid value.
        """
        super().setup_graph()
        self.ax.set_xlabel(f'Time [{time_units}]', fontsize=10)
        error_msg = ('The value {} is not valid for the parameter {} of '
                     'Solution.setup_graph().')
        if xlim in ('time', 'unbounded'):
            self._set_xlim(xlim)
        else:
            raise ValueError(error_msg.format(xlim, 'xlim'))
        if ylim in ('expectations', 'covariances', 'unbounded'):
            self._set_ylim(ylim)
        else:
            raise ValueError(error_msg.format(ylim, 'ylim'))

    def _default_plots(self, expectations=True, variances=True, 
                       covariances=False):
        """Add plots on the default graph."""
        super()._default_plots(one=expectations, symmetric=variances, 
                               nonsymmetric=covariances, three=False)

    def _init_colors(self):
        """Initialize the colors associated to state variables."""
        super()._init_colors()
        self._add_colors_items_two()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        super()._init_plot_methods()
        self._plot = {**self.plot, **self._plot_dict_two()}

    def _init_states_dict(self, states):
        """Initialize the attributes associated to state variables."""
        super()._init_states_dict(states)
        p = len(self.config.network.populations)
        transposed_states = np.transpose(states)
        CAA_flat = transposed_states[2*p : int(round(2*p + p*(p+1)/2))]
        CAA = self._unflat_triangle(CAA_flat)
        CRR_flat = transposed_states[2*p + int(round(p*(p+1)/2)) : 2*p + p*(p+1)]
        CRR = self._unflat_triangle(CRR_flat)
        CAR_flat = transposed_states[2*p + p*(p+1) :]
        CAR = CAR_flat.reshape((p,p,len(CAR_flat[0])))
        CAS = - CAA - CAR
        CRS = - CRR - np.transpose(CAR, axes=(1,0,2))
        CSS = - CAS - CRS
        covs = {'CAA': CAA, 'CRR': CRR, 'CSS': CSS, 
                'CAR': CAR, 'CAS': CAS, 'CRS': CRS}
        self._states_dict = {**self._states_dict, **covs}

    def _label_one(self, X, J):
        """Label for the expectation of `X` for the `J`th population."""
        return f'$\\mathcal{{{X}}}_{{{self.config.network.populations[J].ID}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y` for the `J`th and `K`th
        populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K)])
        return f'$\\mathrm{{C}}_{{{XY}}}^{{{IDs}}}$'

    @staticmethod
    def _unflat_triangle(Y):
        """Reshape a flatten symmetric matrix of vectors into a square one.

        Reshape a flatten symmetric matrix of vectors from a two-dimensional
        array, which contains only the upper triangle of the symmetric square
        matrix of vectors. 

        Parameters
        ----------
        Y : array_like
            The two-dimensional array to reshape into a square array of vectors. 

        Returns
        -------
        array_like
            The array correctly reshaped.
        """
        p = int(round((-1 + np.sqrt(1 + 8*len(Y))) / 2))
        new_array = np.zeros((p,p,len(Y[0])))
        new_array[np.triu_indices(p)] = Y
        new_array[np.tril_indices(p, k=-1)] = new_array[np.triu_indices(p, k=1)]
        return new_array


class SolutionOne(ResultOne, Solution):
    """Represent solutions of the reduced dynamical system for one population.

    The `SolutionOne` class combines the `ResultOne` and `Solution`
    classes to represent solutions obtained from numerical integrations of the
    dynamical system from the extended Wilson--Cowan model, in the case where
    the network has a single population. Its only extensions from the base
    classes are implementation details.

    """

    def _label_one(self, X, J):
        """Label for the expectation of `X`."""
        return f'$\\mathcal{{{X}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y`."""
        return f'$\\mathrm{{C}}_{{{XY}}}$'


class Trajectory(Result):
    """Represent trajectories of the stochastic process.

    The `Trajectory` class extends the `Result` class for the case where the
    result is a possible trajectory of the stochastic process which rules the
    microscopic dynamics of the network. Specifically, it adds methods to plot
    state components and it extends the options to setup a figure. Other changes
    are implementation details, except in methods where it is explicitely noted.

    """

    @classmethod
    def load(cls, ID, name=None, config=None, folder=None):
        """Load the trajectory associated with the ID.

        Load the trajectory obtained when using the configuration of ID `ID`. It
        extends the base class method by loading the `times` array, which is
        assumed to be in a file named *ID - name (times).txt*, where *ID* and
        *name* are indeed `ID` and `name`. Returns `Trajectory` instance.

        See Also
        --------
        Result.load
        """
        name = self._get_name(name)
        filename = _format_filename(folder, ID, f'{name} (times)')
        try:
            times = np.loadtxt(filename, dtype=float)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'It seems that no times array was saved about a result from '
                f'configuration {ID}.') from error
        return super().load(ID, name, config=config, times=times, folder=folder)
        
    def plot_fractions(self):
        """Plot all fractions of populations."""
        self._plot_all_one()

    def setup_graph(self, time_units='ms'):
        """Setup a graphic.

        Setup a graphic started with `Result.initialize_graph`. Extends the base
        class method by setting limits to the figure's axes. 

        Parameters
        ----------
        time_units : str, optional
            Time units to be displayed on the figure. Defaults to `'ms'`, which
            assumes that the transition rates are measured in kHz.
        """
        super().setup_graph()
        self.ax.set_xlabel(f'Time [{time_units}]', fontsize=10)
        self._set_xlim('time')
        self._set_ylim('expectations')

    def _label_one(self, X, J):
        """Label for the fraction `X` of the `J`th population."""
        return f'${X}^{{{self.config.network.populations[J].ID}}}$'


class TrajectoryOne(ResultOne, Trajectory):
    """Represent trajectories of the stochastic process for one population.

    The `TrajectoryOne` class combines the `ResultOne` and `Trajectory`
    classes to represent possible trajectories of the stochastic process which
    rules the microscopic dynamics of the network, in the case where it has only
    one population. Its only extensions from the base classes are implementation
    details.

    """

    def _label_one(self, X, J):
        """Label for the fraction `X` of the population."""
        return f'${X}$'


class Statistics(Result):
    """Represent statistics obtained from sample trajectories.

    The `Statistics` class extends the `Result` class for the case where the
    result is a set of statistics obtained from multiple trajectories of the
    stochastic process which rules the microscopic evolution of the network.

    The most important extension from the `Result` class is a set of methods to
    plot fills between given bounds around a mean value, and methods to plot the
    minimum and maximum values of a variable at each time step. For details,
    see `Statistics.fill`, `Statistics.plot_max` and `Statistics.plot_min`.
    Besides these new methods, it also adds other methods to plot state
    components and it extends the options to setup a figure. Other changes are
    implementation details, except in methods where it is explicitely noted.

    The parameters at initialization are the same as in the base class, except
    that `states` is now expected to be an three-dimensional array of *samples*
    of trajectories of the stochastic process, with time along the first axis,
    state variables along the second, and different simulations along the third.
    Note that this is the format of samples handled by `ChainSimulator`.

    Warns
    -----
    PopNetWarning
        If the given samples do not provide enough trajectories to compute
        unbiased estimates of central moments.

    """

    _default_sample_name = 'Sample'
    """Default name given to samples of trajectories."""

    @classmethod
    def load(cls, ID, sample_name=None, name=None, config=None, folder=None):
        """Load the statistics associated computed from sample trajectories.

        Compute statistics needed to define a `Statistics` instance from loaded
        samples. For each component *X*, the samples are supposed to be in a
        file named *ID - sample_name X.txt*, where *ID* and *sample_name* are
        indeed `ID` and `sample_name`. In the file for a component *X*, it is
        assumed that in each column are the values of *X* with respect to time
        for a given trajectory.

        Parameters
        ----------
        ID : str
            ID of the configuration used to obtain the samples. 
        sample_name : str, optional
            Name associated with the samples to be loaded. Defaults to `None`,
            in which case is it replaced with `'Sample'`.
        name : str, optional
            Name associated with the result. Defaults to `None`, in which case
            it is replaced with `Statistics`.
        config : Configuration, optional
            Configuration to associate with the result. If given, it should have
            the ID `ID`. Defaults to `None`, in which case it is loaded from ID
            `ID`.
        folder : str, optional
            Folder in which the files are located, which should be placed in the
            current directory. Defaults to `None`, in which case the files are
            assumed to be located directly in the current directory.

        Returns
        -------
        Statistics
            Statistics computed from the loaded samples.

        Raises
        ------
        TypeError
            If `config` is neither `None` nor a `Configuration` instance.
        PopNetError
            If `config` has a different ID than `ID`. 
        FileNotFoundError
            If no file is found with the expected name for a component.
        """
        config = self._check_config(config, ID)
        sample_name = self._get_sample_name(sample_name)
        samples = []
        p = len(config.network.populations)
        for J, X in enumerate(config._variables[:2*p]):
            filename = _format_filename(folder, ID, f'{sample_name} {X}')
            try:
                samples.append(np.loadtxt(filename, dtype=float))
            except FileNotFoundError as error:
                raise FileNotFoundError('It seems that no samples have been '
                                        f'saved yet for the component {X} with '
                                        f'configuration {ID}.') from error
        samples = np.transpose(samples, axes=(1,0,2))
        times = np.linspace(config.initial_time, config.final_time, 
                            1 + config.iterations)
        name = self._get_name(name)
        return cls(config, statistics, times, name)

    @property
    def fill(self):
        """Dictionary of methods to add fills for state variables.

        If `X` is a string representing a state variable (that is, either `'A'`,
        `'R'` or `'S'`), then `fill[X]` is a list whose `J`th element is a
        method which adds a fill between two bounds. By default, the bounds are
        given by one standard deviation on each side of the mean value. These
        methods all accept the same arguments as `Statistic.fill_all`.
        """
        return self._fill

    @property
    def plot_max(self):
        """Dictionary of methods to plot maxima of state variables.

        Analogous dictionary as `Result.plot` with keys `'A'`, `'R'` and `'S'`,
        but were the methods plot *maxima* of state variables instead of their
        mean values.
        """
        return self._plot_max

    @property
    def plot_min(self):
        """Dictionary of methods to plot minima of state variables.

        Analogous dictionary as `Result.plot` with keys `'A'`, `'R'` and `'S'`,
        but were the methods plot *maxima* of state variables instead of their
        mean values.
        """
        return self._plot_min
        
    def fill_all(self, bound='std', alpha=.25):
        """Add fills between given bounds.

        Add fills between given bounds around the mean values of all fractions
        of populations.

        Parameters
        ----------
        bound : {'std', 'extrema'}, optional
            Described the bounds between which to fill. If `'std'`, the region
            bounded by one standard deviation around the mean value will be
            filled. If `'extrema'`, the region bounded by the minimum and
            maximum values of the component will be filled. Defaults to `'std'`.
        alpha : float
            Transparency parameter of the fill. Defaults to 0.25.
        """
        self._fill_all(bound, alpha)

    def plot_expectations(self, **kwargs):
        """Plot all expectations of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_one(**kwargs)

    def plot_variances(self, **kwargs):
        """Plot all variances of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_two(symmetric=True, nonsymmetric=False, **kwargs)

    def plot_covariances(self, **kwargs):
        """Plot all non-symmetric covariances of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_two(symmetric=False, nonsymmetric=True, **kwargs)

    def plot_third_moments(self, **kwargs):
        """Plot all third central moments of *A*'s, *R*'s and *S*'s.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_three(**kwargs)

    def setup_graph(self, time_units='ms', xlim='time', ylim='expectations'):
        """Setup a graphic.

        Setup a graphic started with `Result.initialize_graph`. Extends the base
        class method by setting limits to the figure's axes.

        Parameters
        ----------
        time_units : str, optional
            Time units to be displayed on the figure. Defaults to `'ms'`, which
            assumes that the transition rates are measured in kHz.
        xlim : {'time', 'unbounded'}, optional
            Decides how the horizontal axis is bounded. If `'time'`, it will be
            bounded by the initial and final times of the configuration. If
            `'unbounded'`, it will not be bounded. Defaults to `'time'`.
        ylim : {'expectations', 'covariances', 'unbounded'}, optional
            Decides how the vertical axis is bounded. If `'expectations'`, it
            will be bounded between 0 and 1. If `'covariances'`, it will be
            bounded between -1/4 and 1. If `'unbounded'`, it will not be
            bounded. Defaults to `'expectations'`.

        Raises
        ------
        ValueError
            If `xlim` or `ylim` is given a non-valid value.
        """
        super().setup_graph()
        self.ax.set_xlabel(f'Time [{time_units}]', fontsize=10)
        error_msg = ('The value {} is not valid for the parameter {} of '
                     'Statistics.setup_graph().')
        if xlim in ('time', 'unbounded'):
            self._set_xlim(xlim)
        else:
            raise ValueError(error_msg.format(xlim, 'xlim'))
        if ylim in ('expectations', 'covariances', 'unbounded'):
            self._set_ylim(ylim)
        else:
            raise ValueError(error_msg.format(ylim, 'ylim'))

    def _default_plots(self, expectations=True, variances=True,
                       covariances=False, third_moments=False):
        """Add plots on the default graph."""
        super()._default_plots(one=expectations, symmetric=variances, 
                               nonsymmetric=covariances, three=third_moments)

    def _fill_all(self, bound, alpha):
        """Add fills for all *A*'s, *R*'s and *S*'s."""
        p = len(self.config.network.populations)
        for J in range(p):
            for X in ['A', 'R', 'S']:
                self.fill[X][J](bound=bound, alpha=alpha)

    def _fill_dict(self):
        """Return a dictionary of methods to add fills for each population."""
        p = len(self.config.network.populations)
        return {X: [self._make_fill(X, J) for J in range(p)]
                for X in ['A', 'R', 'S']}

    @staticmethod
    def _get_central_moment(samples, stacklevel=1):
        """Compute a central moment from samples.

        Compute a central moment from samples. The axes of `samples` are assumed
        to be ordered in the same way as those handled by `ChainSimulator`, but
        the second axis is assumed to span only the relevant components. Hence,
        if the second axis has length *k*, the central moment computed will be
        of order *k*.
        """
        T = len(samples)
        order = len(samples[0])
        executions = len(samples[0,0])
        coeff = 1 / executions
        if order == 2:
            if executions > 1:
                coeff = 1 / (executions - 1)
            else:
                warn('Not enough executions to compute an unbiased estimate of '
                     'a covariance. A biased one is computed instead.',
                     category=PopNetWarning, stacklevel=stacklevel)
        elif order == 3:
            if executions > 2:
                coeff = executions / ((executions - 1) * (executions - 2))
            else:
                warn('Not enough executions to compute an unbiased estimate of '
                     'a third central moment. A biased one is computed '
                     'instead.', category=PopNetWarning, stacklevel=stacklevel)
        means = np.mean(samples, axis=2)
        means = np.resize(means, (executions, T, order)).transpose(1,2,0)
        prod = np.prod(samples - means, axis=1)
        return coeff * np.sum(prod, axis=1)

    @classmethod
    def _get_sample_name(cls, sample_name):
        """Return `sample_name`, or the default sample name if it is `None`."""
        if sample_name is None:
            return cls._default_sample_name
        else:
            return sample_name

    def _init_colors(self):
        """Initialize the colors associated to state variables."""
        super()._init_colors()
        self._add_colors_items_two()
        self._add_colors_items_three()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        super()._init_plot_methods()
        self._plot = {**self.plot, **self._plot_dict_two(), 
                      **self._plot_dict_three()}
        self._plot_min = self._plot_dict_one(states=self._min_dict, ls='--',
                                             lw=1, label_func=self._label_min)
        self._plot_max = self._plot_dict_one(states=self._max_dict, ls='--',
                                             lw=1, label_func=self._label_max)
        self._fill = self._fill_dict()

    def _init_states_dict(self, samples):
        """Initialize the state variables dictionary."""
        p = round(len(samples[0]) / 2)
        samples_S = 1 - samples[:,:p] - samples[:,p:]
        samples = np.concatenate((samples, samples_S), axis=1)
        zero = {'A': 0, 'R': p, 'S': 2*p}

        self._min_dict = {X: [np.min(samples[:,zero[X]+J], axis=1) 
                              for J in range(p)]
                          for X in ['A', 'R', 'S']}
        self._max_dict = {X: [np.max(samples[:,zero[X]+J], axis=1) 
                              for J in range(p)]
                          for X in ['A', 'R', 'S']}
        expect = {X: [np.mean(samples[:,zero[X]+J], axis=1) for J in range(p)]
                  for X in ['A', 'R', 'S']}

        def element(CXY, J, K):
            C, X, Y = CXY
            return self._get_central_moment(samples[:,[zero[X]+J, zero[Y]+K]], 
                                            stacklevel=9+int(p==1))

        cov = {CXY: [[element(CXY, J, K) for K in range(p)] for J in range(p)]
              for CXY in ['CAA', 'CRR', 'CSS', 'CAR', 'CAS', 'CRS']}

        def has_to_be_set(X, Y, Z, J, K, L):
            if X == Y == Z:
                return J <= K <= L
            if X == Y:
                return J <= K
            if Y == Z:
                return K <= L
            return True

        def element(XYZ, J, K, L):
            X, Y, Z = XYZ
            if has_to_be_set(X, Y, Z, J, K, L):
                return self._get_central_moment(
                        samples[:,[zero[X]+J, zero[Y]+K, zero[Z]+L]], 
                        stacklevel=10+int(p==1))
            return None

        triplets = ['AAA', 'AAR', 'AAS', 'ARR', 'ARS', 
                    'ASS', 'RRR', 'RRS', 'RSS', 'SSS']
        thirds = {XYZ: [[[element(XYZ, J, K, L) for L in range(p)]
                         for K in range(p)] for J in range(p)]
                  for XYZ in triplets}

        self._states_dict = {**expect, **cov, **thirds}

    def _label_max(self, X, J):
        """Label for the maximum of `X` for the `J`th population."""
        pop = self.config.network.populations[J]
        return f'$\\mathrm{{max}}\\, {{{X}}}_{{{pop.ID}}}$'

    def _label_min(self, X, J):
        """Label for the minimum of `X` for the `J`th population."""
        pop = self.config.network.populations[J]
        return f'$\\mathrm{{min}}\\, {{{X}}}_{{{pop.ID}}}$'

    def _label_one(self, X, J):
        """Label for the expectation of `X` for the `J`th population."""
        return f'$\\mathcal{{{X}}}_{{{self.config.network.populations[J].ID}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y` for the `J`th and `K`th
        populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K)])
        return f'$\\mathrm{{C}}_{{{XY}}}^{{{IDs}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the third central moment for variables `X`, `Y` and `Z` for
        the `J`th, `K`th and `L`th populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K, L)])
        return f'$\\mathrm{{M}}_{{{XYZ}}}^{{{IDs}}}$'

    def _make_fill(self, X, J):
        """Define the method to add a fill around the fraction variable `X` for
        the `J`th population."""
        def f(bound='std', alpha=.25):
            self._check_if_initialized()
            if bound == 'std':
                CXX = f'C{X}{X}'
                low = (self._states_dict[X][J] 
                        - np.sqrt(self._states_dict[CXX][J][J]))
                high = (self._states_dict[X][J] 
                        + np.sqrt(self._states_dict[CXX][J][J]))
            elif bound == 'extrema':
                low = self._min_dict[X][J]
                high = self._max_dict[X][J]
            fill = self.ax.fill_between(self.times, low, high, 
                                        color=self.colors[X][J], alpha=alpha)
            return fill
        return f


class StatisticsOne(ResultOne, Statistics):
    """Represent statistics got from sample trajectories for one population.

    The `StatisticsOne` class combines the `ResultOne` and `Statistics`
    classes to represent statistics obtained from multiple trajectories of the
    stochastic process, in the case where the network has a single population.
    Its only extensions from the base classes are implementation details.

    """
        
    def _fill_all(self, bound, alpha):
        """Add fills for all *A*, *R* and *S*."""
        for X in ['A', 'R', 'S']:
            self.fill[X](bound=bound, alpha=alpha)

    def _fill_dict(self):
        """Return a dictionary of methods to add a fill."""
        return {X: self._make_fill(X, 0) for X in ['A', 'R', 'S']}

    def _label_max(self, X, J):
        """Label for the maximum of `X`."""
        return f'$\\mathrm{{max}}\\, {{{X}}}$'

    def _label_min(self, X, J):
        """Label for the minimum of `X`."""
        return f'$\\mathrm{{min}}\\, {{{X}}}$'

    def _label_one(self, X, J):
        """Label for the expectation of `X`."""
        return f'$\\mathcal{{{X}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y`."""
        return f'$\\mathrm{{C}}_{{{XY}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the third central moment between `X`, `Y` and `Z`."""
        return f'$\\mathrm{{M}}_{{{XYZ}}}$'


class Spectrum(Result):
    """Represent spectra of other results.

    The `Spectrum` class extends the `Result` class for the case where the
    result is the spectrum of another result. Specifically, it defines methods
    to plot the spectra of state components, and it extends the options to setup
    a figure. Its data attributes are the same as in the base class, but here
    `times` is replaced with `freqs`; see `Spectrum.freqs`. Finally, the
    `Spectrum` class suppresses `Result`'s `load` and `get_spectrum` methods.

    The recommended way of initializing a `Spectrum` instance is from another
    `Result` instance, with `Result.get_spectrum`. Parameters at initialization
    are a bit different here than in the base class, so they are listed again.

    Parameters
    ----------
    config : Configuration
        The configuration used to obtain the result.
    states : dict of array_like
        Dictionary in which to each state component is associated an array. Such
        an array should give the values, for each combination of populations,
        of the state component with respect to time. This is the format in which
        data is kept internally in `Result` classes.
    times : array_like
        An array representing time.
    source : str
        The name of the class from which comes the spectrum.
    name : str, optional
        A name associated to the result. Defaults to `None`, in which case it is
        replaced with the name of the class.

    """

    def __init__(self, config, states, times, source, name=None):
        if source in ('Solution', 'Trajectory', 'Statistics', 'Result'):
            self._source = source
        else:
            raise ValueError(f'Unknown source {source} for Spectrum instance.')
        super().__init__(config, states, times, name)

    @classmethod
    def load(cls):
        raise AttributeError('\'Spectrum\' object has no attribute \'load\'')

    @property
    def freqs(self):
        """Frequencies.

        Frequencies for which the Fourier transforms gives the amplitudes.
        Replaces `Result.times`.
        """
        return self.times

    @freqs.setter
    def freqs(self, new_value):
        self.times = new_value

    def get_spectrum(self):
        raise AttributeError('\'Spectrum\' object has no attribute '
                             '\'get_spectrum\'')

    def plot_spectra(self, **kwargs):
        """Plot spectra for all fractions of populations.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.
        """
        self._plot_all_one(**kwargs)

    def plot_variances_spectra(self, **kwargs):
        """Plot spectra of all variances.

        Plot spectra of all variances of *A*'s, *R*'s and *S*'s, if such
        variances are defined.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.

        Raises
        ------
        PopNetError
            If no variances are defined for this result.
        """
        if self._source in ('Solution', 'Statistics'):
            self._plot_all_two(symmetric=True, nonsymmetric=False, **kwargs)
            return
        raise PopNetError('No variances defined for this result.')

    def plot_covariances_spectra(self, **kwargs):
        """Plot spectra of all non-symmetric covariances.

        Plot spectra of all non-symmetric covariances of *A*'s, *R*'s and *S*'s,
        if such covariances are defined.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.

        Raises
        ------
        PopNetError
            If no covariances are defined for this result.
        """
        if self._source in ('Solution', 'Statistics'):
            self._plot_all_two(symmetric=False, nonsymmetric=True, **kwargs)
            return
        raise PopNetError('No covariances defined for this result.')

    def plot_third_moments_spectra(self, **kwargs):
        """Plot spectra of all third central moments.

        Plot spectra of all third central moments of *A*'s, *R*'s and *S*'s, if
        such moments are defined.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to all methods to plot components,
            that is, to the `plot` method of a [`matplotlib.axes.Axes`](
            https://matplotlib.org/3.3.3/api/axes_api.html#matplotlib.axes.Axes)
            object.

        Raises
        ------
        PopNetError
            If no third central moments are defined for this result.
        """
        if self._source == 'Statistics':
            self._plot_all_three(**kwargs)
            return
        raise PopNetError('No third central moments defined for this result.')

    def setup_graph(self, freq_units='kHz', yscale='linear', xlim='freqs'):
        """Setup a graphic.

        Setup a graphic started with `Result.initialize_graph`. Extends the base
        class method by labeling and setting limits to the horizontal axis, and
        by allowing the option to set the scale of the vertical axis.

        Parameters
        ----------
        freq_units : str, optional
            Frequency units to be displayed on the figure. They should be those
            in which the transition rates are measured. Defaults to `'kHz'`.
        yscale : {'linear', 'log'}, optional
            Defines the scale of the vertical axis. Defaults to `'linear'`.
        xlim : {'freqs', 'config', 'unbounded'}, optional
            Decides how the horizontal axis is bounded. If `'freqs'`, it will be
            bounded between 0 and the highest frequency. If `'config'`, it will
            be bounded between 0 and the highest frequency obtained from the
            times array given by the configuration. If `'unbounded'`, it will
            not be bounded. Defaults to `'freqs'`.

        Raises
        ------
        ValueError
            If `xlim` is given a non-valid value.
        """
        super().setup_graph()
        self.ax.set_yscale(yscale)
        self.ax.set_xlabel(f'Frequency [{freq_units}]', fontsize=10)
        if xlim in ('freqs', 'config', 'unbounded'):
            self._set_xlim(xlim)
        else:
            raise ValueError(f'The value {xlim} is not valid for the parameter '
                             'xlim of Spectrum.setup_graph().')

    def _default_plots(self, expectations=True, variances=True, 
                       covariances=False, third_moments=False):
        """Add plots on the default graph."""
        sym = self._source in ('Solution', 'Statistics') and variances
        nonsym = self._source in ('Solution', 'Statistics') and covariances
        three = self._source == 'Statistics' and third_moments
        super()._default_plots(one=expectations, symmetric=sym, 
                               nonsymmetric=nonsym, three=three)

    def _init_abscissa(self, times):
        """Initialize array related to the independant variable."""
        size = round(len(times)/2+.8)
        # If n is the length of the times array, the size of freqs has to be
        # n/2 + 1 if n is even, and (n+1)/2 if n is odd.
        self._times = np.linspace(0, len(times) / times[-1], size)

    def _init_colors(self):
        """Initialize the colors associated to state variables."""
        super()._init_colors()
        if self._source in ('Solution', 'Statistics'):
            self._add_colors_items_two()
        if self._source == 'Statistics':
            self._add_colors_items_three()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        super()._init_plot_methods()
        if self._source in ('Solution', 'Statistics'):
            self._plot = {**self._plot, **self._plot_dict_two()}
        if self._source == 'Statistics':
            self._plot = {**self._plot, **self._plot_dict_three()}

    def _init_states_dict(self, states):
        """Initialize the state variables dictionary."""
        p = len(self.config.network.populations)
        self._states_dict = dict.fromkeys(states)
        axes = {'A': 1, 'R': 1, 'S': 1, 'CAA': 2, 'CRR': 2, 'CSS': 2, 'CAR': 2,
                'CAS': 2, 'CRS': 2, 'AAA': 3, 'AAR': 3, 'AAS': 3, 'ARR': 3,
                'ARS': 3, 'ASS': 3, 'RRR': 3, 'RRS': 3, 'RSS': 3, 'SSS': 3}
        for key in states:
            if key in axes:
                transform = np.fft.rfft(states[key], axis=axes[key])
                self._states_dict[key] = np.abs(transform)
            else:
                raise ValueError(f'{key} is not a valid state variable.')

    def _label_one(self, X, J):
        """Label for the spectrum of `X` for the `J`th population."""
        J_ID = self.config.network.populations[J].ID
        if self._source in ('Solution', 'Statistics'):
            return f'$\\hat{{\\mathcal{{{X}}}}}_{{{J_ID}}}$'
        elif self._source == 'Trajectory':
            return f'$\\hat{{{X}}}_{{{J_ID}}}$'

    def _label_two(self, XY, J, K):
        """Label for the spectrum of the covariance between `X` and `Y` for the
        `J`th and `K`th populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K)])
        return f'$\\hat{{\\mathrm{{C}}}}_{{{XY}}}^{{{IDs}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the spectrum of the third central between `X`, `Y` and `Z`
        for the `J`th, `K`th and `L`th populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K, L)])
        return f'$\\hat{{\\mathrm{{M}}}}_{{{XYZ}}}^{{{IDs}}}$'


class SpectrumOne(Spectrum, ResultOne):
    """Represent spectra of other results for one population.

    The `SpectrumOne` class combines the `ResultOne` and `Solution`
    classes to represent spectra of other results, in the case where the network
    has a single population. Its only extensions from the base classes are
    implementation details.

    """

    def _label_one(self, X, J):
        """Label for the spectrum of `X`."""
        if self._source in ('Solution', 'Statistics'):
            return f'$\\hat{{\\mathcal{{{X}}}}}$'
        elif self._source == 'Trajectory':
            return f'$\\hat{{{X}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y` for the `J`th and `K`th
        populations."""
        return f'$\\hat{{\\mathrm{{C}}}}_{{{XY}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the third central moment between `X`, `Y` and `Z`."""
        return f'$\\hat{{\\mathrm{{M}}}}_{{{XYZ}}}$'


def build_network(ID, matrix):
    """Get a network from a weight matrix.

    Get a network with a weight matrix given by `matrix`.

    Parameters
    ----------
    ID : str
        ID given to the network.
    matrix : array_like
        Weight matrix specifying the connections between neurons in the network.
        It has to be square and to have real entries.

    Returns
    -------
    MicroNetwork
        Network initialized with weight matrix corresponding to `matrix`.
    """
    N = matrix.shape[0]
    if matrix.shape != (N, N):
        raise ValueError('The given matrix should be square.')
    net = default_network(ID, scale='micro')
    net.c = N * np.mean(matrix)
    net.W = matrix
    return net

def default_config(ID, scale='macro'):
    """Define a configuration with default parameters.

    Define a new configuration with default parameters and a given ID. The
    network associated to this configuration will be defined with
    `default_network`, using the same `scale`.

    Parameters
    ----------
    ID : str
        ID given to the new configuration. Its first character should be a
        positive integer, which is taken to be the number of populations.
    scale : {'macro', 'micro'}, optional
        Determine whether the new network has a defined microscopic structure.
        If `'micro'`, a default size of 100 will be given to each population. If
        `'macro'`, population sizes will remain undefined.

    Returns
    -------
    Configuration, MicroConfiguration, ConfigurationOne or MicroConfigurationOne
        The configuration with default parameters. It will be a subclass of
        `Configuration` if it is more appropriate according to `scale` and to
        the number of populations of the network.
    """
    net = default_network(ID, scale=scale)
    return make_config(net, ID)


def default_network(ID, scale='macro'):
    """Define a network with default parameters.
    
    Define a new network with default parameters and a given ID.

    Parameters
    ----------
    ID : str
        ID given to the new network. Its first character should be a positive
        integer, which is taken to be the number of populations.
    scale : {'macro', 'micro'}, optional
        Determine whether the network has a defined microscopic structure. If
        `'micro'`, a default size of 100 will be given to each population. If
        `'macro'`, population sizes will remain undefined.

    Returns
    -------
    Network or MicroNetwork
        The network with default parameters. It will be a `MicroNetwork`
        instance if the sizes of the populations are defined.

    Raises
    ------
    PopNetError
        If a non-valid value is passed to `scale`.
    """
    p = int(ID[0])
    if p == 1:
        pops = [Population('Population')]
    else:
        pops = [Population(f'Population {j+1}', ID=str(j+1)) for j in range(p)]
    if scale == 'macro':
        return Network(ID, pops)
    elif scale == 'micro':
        for pop in pops:
            pop.size = 100
        return MicroNetwork(ID, pops)
    raise PopNetError(f'Unknown scale {scale} to create a network.')


def get_integrator(config, system='extended'):
    """Define an integrator.

    Define an integrator using the most appropriate class constructor
    according to the number of populations of the network. 

    Parameters
    ----------
    config : Configuration
        Configuration to associate to the integrator.
    system : {'extended', 'simple'}, optional
        Decides which dynamical system is integrated. If `'extended'`, it will
        be that of the extended Wilson--Cowan model. If `'simple'`, it will be
        that of the Wison--Cowan model with refractory state explicitely
        considered. Defaults to `'extended'`.

    Returns
    -------
    SimpleIntegrator, ExtendedIntegrator or ExtendedIntegratorOne
        Integrator initialized with given parameters.

    Raises
    ------
    PopNetError
        If `system` is given a non-valid value.
    """
    if system == 'extended':
        if config.ID[0] == '1':
            return ExtendedIntegratorOne(config)
        return ExtendedIntegrator(config)
    elif system == 'simple':
        return SimpleIntegrator(config)
    raise PopNetError(f'Unknown dynamical system {system}.')


def get_simulator(config, mode='individual'):
    """Define a simluator.

    Define a simulator in order to perform either individual simulations, or
    chains of simulations.

    Parameters
    ----------
    config : Configuration
        Configuration to associate to the simulator.
    mode : {'individual', 'chain'}, optional
        How the simulations should be executed. If `'individual'`, the simulator
        will be defined to run one simulation at a time. If `'chain'`, it will
        be defined to run a sequence of simulations at every run.

    Returns
    -------
    SimpleSimulator or ChainSimulator
        Simulator initialized with given configuration.
    """
    if mode == 'individual':
        return SimpleSimulator(config)
    elif mode == 'chain':
        return ChainSimulator(config)
    raise PopNetError(f'Unknown execution mode {mode}.')


def load_config(load_ID, new_ID=None, network=None, folder=None):
    """Load a configuration from a text file.

    Load the configuration parameters from a text file. This text file is
    expected to be named *ID - Configuration.txt* (with *ID* replaced with
    the configuration's actual ID). When reading it as a single string, it
    is also expected to have the format of a string representation of a
    `Configuration` instance. Note that this is the format of text file
    saved by `Configuration.save`.

    Parameters
    ----------
    load_ID : str
        ID of the configuration to load. 
    new_ID : str, optional
        ID of the configuration to create with the parameters of `load_ID`.
        Defaults to `None`, in which case `load_ID` is used. 
    network : Network, optional
        The network to associate with the configuration. Defaults to `None`,
        in which case a new `Network` instance is created with `load_network`.
    folder : str, optional
        Folder in which the text file is located. It should be placed in the
        current directory. Defaults to `None`, and the file is assumed to be
        located directly in the current directory in that case.

    Returns
    -------
    Configuration, MicroConfiguration, ConfigurationOne or MicroConfigurationOne
        The loaded configuration with ID `new_ID`. It will be a subclass of
        `Configuration` if it is more appropriate according to the number of
        populations of the network and to their sizes.

    Raises
    ------
    FileNotFoundError
        If no file is found with the expected name.
    PopNetError
        If the file contains inconsistent information or unexpected
        parameters to set.
    FormatError
        If the file does not have the expected format.

    Warns
    -----
    PopNetWarning
        If the information in the file is inconsistent.
    """
    filename = _format_filename(folder, load_ID, 'Configuration')
    if new_ID is None:
        new_ID = load_ID
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError as error:
        raise FileNotFoundError('No file is available to load configuration '
                                f'{load_ID}. Maybe the configuration has not '
                                'been saved, or maybe the file containing the '
                                'data does not have the expected name. It '
                                'should have the format ID - Configuration.txt.'
                                ) from error
    config = _read_config_file(load_ID, new_ID, network, folder, lines)
    if config is None:
        raise FormatError(f'It seems that the file {load_ID} - Configuration.txt'
                          ' does not have the correct format to import the conf'
                          f'iguration {load_ID}. It should have the format of a'
                          ' string representation of a Configuration instance.')
    return config


def load_network(load_ID, new_ID=None, folder=None):
    """Load a network from a text file.

    Define a new network from parameters in a text file. This file is expected
    to be named *ID - Network parameters.txt* where *ID* is the network's actual
    ID. If the file is read as a single string, it is expected to have the
    format of a string representation of a `Network` instance. Note that this is
    the format of text file saved by `Network.save`.

    Parameters
    ----------
    load_ID : str
        ID of the network to load. 
    new_ID : str, optional
        ID of the network to define. Defaults to `None`, in which case
        `load_ID` is used. 
    folder : str, optional
        Folder in which the text file is located. If given, if should be in
        the current directory. Defaults to `None`, in which case the text
        file is expected to be in the current directory.

    Returns
    -------
    Network or MicroNetwork
        The loaded network. It will be a `MicroNetwork` if a size is given for
        every population.

    Raises
    ------
    FileNotFoundError
        If no file is found with the expected name.
    PopNetError
        If the information in the file is not consistent with `load_ID`.
    FormatError
        If the file does not have the expected format.
    """
    filename = _format_filename(folder, load_ID, 'Network parameters')
    if new_ID is None:
        new_ID = load_ID
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError as error:
        raise FileNotFoundError(f'No file is available to load network {load_ID}'
                                '. Maybe no parameters have been saved, or maybe'
                                ' the file containing the data does not have the'
                                f' expected name. It should be {load_ID} - '
                                'Network parameters.txt.') from error
    if load_ID != (other_ID := lines[0].strip().split()[-1]):
        raise PopNetError(f'The file {load_ID} - Network parameters.txt seems '
                          f'to contain information about a network {other_ID} '
                          f'rather than {load_ID}; PopNet is confused.')
    j = 2
    populations = []
    while j < len(lines):
        if lines[j].startswith('Connection matrix'):
            break
        # At index j+1 a population's description starts. We loop until the
        # end of the description, that is, when an empty line is reached.
        k = j + 1
        while lines[k] != '\n':
            k += 1
        populations.append(Population._load(lines[j : k]))
        j = k + 1
    else:
        raise FormatError(f'It seems that the file {load_ID} - Network param'
                          'eters.txt does not have the correct format to import'
                          f' the network {load_ID}. It should have the format '
                          'of a string representation of a Network instance.')
    if any(pop.size is None for pop in populations):
        net = Network(new_ID, populations)
    else:
        net = MicroNetwork(new_ID, populations)
    # Set the connection matrix from the file's data.
    string = ''.join(lines[j+1 : j+1+len(net.populations)])
    net._set_c_from_string(string)
    return net


def make_config(network, ID=None, **kwargs):
    """Define a configuration.

    Define a new configuration using the most appropriate class constructor
    according to the number of populations of the network and to the type of
    the network.

    Parameters
    ----------
    network : Network
        The network used with this configuration.
    ID : str, optional
        The identificator of the configuration. The default is to take the
        network's ID.
    **kwargs
        Keyword arguments to be passed to the class constructor.

    Returns
    -------
    Configuration, MicroConfiguration, ConfigurationOne or MicroConfigurationOne
        A configuration initialized with `ID` and `network`. It will be a
        subclass of `Configuration` if it is more appropriate according to the
        number of populations of the network and to their sizes.
    """
    if ID is None:
        ID = network.ID
    if isinstance(network, MicroNetwork) and ID[0] == '1':
        return MicroConfigurationOne(network, ID=ID, **kwargs)
    if isinstance(network, MicroNetwork):
        return MicroConfiguration(network, ID=ID, **kwargs)
    if ID[0] == '1':
        return ConfigurationOne(network, ID=ID, **kwargs)
    return Configuration(network, ID=ID, **kwargs)


def init_standard_graph(colors=True, usetex=False, figsize=(5,3.75), dpi=150, 
                        **kwargs):
    """Initialize a graphic.

    Create a `matplotlib` figure with default formatting.

    Parameters
    ----------
    colors : bool, optional
        Determines if the default color cycle is changed. Defaults to `True`.
    usetex : bool, optional
        Determines if TeX is used in the figure. If it is used, the LaTeX
        packages `newpxtext` and `newpxmath` are also loaded to use the font
        Palatino. Defaults to `False`.
    figsize : tuple of float, optional
        Width and height of the figure in inches. Defaults to `(5, 3.75)`.
    dpi : int
        Resolution of the figure in dots per inches. Defaults to 150.
    **kwargs
        Keywords arguments to be passed to [`matplotlib.pyplot.figure`](
        https://tinyurl.com/plt-figure). 

    Returns
    -------
    fig : matplotlib.figure.Figure
        The initialized `matplotlib` figure.
    ax : matplotlib.axes.Axes
        The axes of `fig`.
    """
    if usetex:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = ('\\usepackage{newpxtext}'
                                               '\\usepackage{newpxmath}')
    else:
        mpl.rcParams['font.serif'] = ['Palatino Linotype']
        mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        (150/255,10/255,47/255), 'midnightblue', 'goldenrod', 
        'seagreen', 'purple'])
    fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
    ax = fig.gca()
    ax.tick_params(direction='in', top=True, right=True)
    fig.set_tight_layout(True)
    return fig, ax


def _format_filename(folder, ID, name, extension='txt'):
    """Format filenames used in the module."""
    if folder is None:
        return f'{ID} - {name}.{extension}'
    else:
        return f'{folder}/{ID} - {name}.{extension}'


def _read_config_file(load_ID, new_ID, network, folder, lines):
    """Read a configuration file given as a list of lines.

    If the file does not have the expected format, return `None`. Else, return
    loaded configuration. See `load_config` for details.
    """
    if not lines[0].startswith('Configuration'):
        return None
    if load_ID != (other_ID := lines[0].strip().split()[-1]):
        raise PopNetError(f'The file {load_ID} - Configuration.txt seems to cont'
                          f'ain the information about a configuration {other_ID}'
                          f'rather than {load_ID}; PopNet is confused.')
    if not lines[2].startswith('Network'):
        return None
    # If no network was given, load it from the ID given in the file.
    if network is None:
        network_ID = lines[2].strip().split()[-1]
        network = load_network(network_ID, folder=folder)
    # Define the configuration.
    config = make_config(network, ID=new_ID)
    if not lines[4].startswith('Parameters'):
        return None
    # Load parameters. 
    params = {'initial_time': None, 'final_time': None, 
              'iterations': None, 'delta': None, 'executions': None}
    for j in range(5, len(lines)):
        if lines[j].startswith('Input'):
            break
        words = lines[j].strip().split()
        if len(words) == 0:
            continue
        elif len(words) == 2:
            param = words[1]
            val = words[0]
        elif len(words) == 3:
            param = words[0].replace('ti', 'initial_time')
            param = param.replace('tf', 'final_time')
            param = param.replace(u'\u0394t', 'delta')
            val = words[2]
        else:
            raise FormatError(f'Unexpected line {lines[j]} in saved file for '
                              f'configuration {load_ID}.')
        if param in params:
            params[param] = val
        elif param == 'execution':
            params['executions'] = val
        else:
            raise PopNetError(f'Unexpected parameter {param} to load with '
                              f'configuration {load_ID}')
    else:
        # If the loop ended without breaking, something went wrong.
        return None
    for param in params:
        if param == 'delta' and params['iterations'] is not None:
            if params[param] not in (None, str(config.delta)):
                warn(f'The time interval given in the file {load_ID} - Configu'
                     f'ration.txt is {params[param]}. However, with the initial'
                     f' time of {config.initial_time}, the final time of '
                     f'{config.final_time}, and {config.iterations} iterations,'
                     f' the time interval should have been {config.delta}. The '
                     'value given in the file has been replaced by the latter.', 
                     category=PopNetWarning, stacklevel=3)
                continue
        if params[param] is not None:
            setattr(config, param, params[param])
    # Set the input value.
    string_Q = lines[j+1].split('=')[-1].strip()
    config.Q = ast.literal_eval(re.sub(r'\s+', ',', string_Q))
    if not lines[j+3].startswith('Initial state'):
        return None
    # Set the initial state.
    state = []
    for line in lines[j+4 : j+4+len(config.initial_state)]:
        state.append(float(line.strip().split()[-1]))
    config.initial_state = state
    return config
