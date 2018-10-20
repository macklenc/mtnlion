"""
Equation solver
"""
import logging
from typing import Union, Dict, List, Callable

import munch
import numpy as np

from . import ldp

logger = logging.getLogger(__name__)


def rmse(estimated: np.ndarray, true: np.ndarray) -> Union[np.ndarray, None]:
    """
    Calculate the root-mean-squared error between two arrays
    :param estimated: estimated solution
    :param true: 'true' solution
    :return: root-mean-squared error
    """
    return np.sqrt(((estimated - true) ** 2).mean(axis=1))


def fetch_params(filename: str) -> Union[Dict[str, Dict[str, float]], None, munch.DefaultMunch]:
    """TODO: read template from config file"""
    print('Loading Cell Parameters')
    params = dict()
    sheet = ldp.read_excel(filename, 0)
    (ncol, pcol) = (2, 3)
    params['const'] = ldp.load_params(sheet, range(7, 15), ncol, pcol)
    params['neg'] = ldp.load_params(sheet, range(18, 43), ncol, pcol)
    params['sep'] = ldp.load_params(sheet, range(47, 52), ncol, pcol)
    params['pos'] = ldp.load_params(sheet, range(55, 75), ncol, pcol)

    return munch.DefaultMunch.fromDict(params)


def find_ind(data: np.ndarray, value: Union[List[int], List[float]]) -> np.ndarray:
    """
    Find the indices of the values given in the data.
    :param data: data to find indices in
    :param value: values to find indices with
    :return: indices of value in data
    """
    return np.nonzero(np.in1d(data, value))[0]


def find_ind_near(data: np.ndarray, value: Union[List[int], List[float]]) -> np.ndarray:
    """
    Find the indices of the closest values given in the data.
    :param data: data to find indices in
    :param value: values to find indices with
    :return: indices of value in data
    """

    return np.array([(np.abs(data - v)).argmin() for v in value])


# TODO: move subdomain logic here
class Mountain:
    """
    Container for holding n-variable n-dimensional data in space and time.
    """

    def __init__(self, mesh: np.ndarray, pseudo_dim_mesh: np.ndarray, time_mesh: np.ndarray,
                 boundaries: Union[np.ndarray, List[float]], **kwargs: np.ndarray) -> None:
        """
        Store the solutions to each given parameter

        :param mesh: Solution mesh
        :param boundaries: internal boundaries in the mesh
        :param kwargs: arrays for each solution
        """

        logger.info('Initializing solution data...')
        self.data = munch.Munch(kwargs)
        self.mesh = mesh
        self.pseudo_mesh = pseudo_dim_mesh
        self.time_mesh = time_mesh
        self.boundaries = boundaries

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Retrieve dictionary of Mountain to serialize
        :return: data dictionary
        """
        d = {'mesh': self.mesh, 'pseudo_mesh': self.pseudo_mesh, 'time_mesh': self.time_mesh,
             'boundaries': self.boundaries}
        return dict(d, **self.data)

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> 'Mountain':
        """
        Convert dictionary to SolutionData
        :param data: dictionary of formatted data
        :return: consolidated simulation data
        """
        return cls(data.pop('mesh'), data.pop('pseudo_mesh'), data.pop('time_mesh'), data.pop('boundaries'), **data)

    def filter(self, index: Union[List['ellipsis'], List[int], List[slice], slice], func: Callable = lambda x: x) \
            -> Dict[str, np.ndarray]:
        """
        Filter through dictionary to collect sections of the contained ndarrays.
        :param index: subset of arrays to collect
        :param func: function to call on every variable in data
        :return: dictionary of reduced arrays
        """

        return {k: func(v[index]) for k, v in self.data.items() if not np.isscalar(v)}

    def filter_time(self, index: Union[List['ellipsis'], List[int], slice], func: Callable = lambda x: x) -> 'Mountain':
        """
        Filter the Mountain for a subset of time indices. For example::
            solution.filter_time(slice(0,5))
            solution.filter_time([0, 3, 5])
            solution.filter_time(slice(step=-1))
            solution.filter_time(numpy.where(solution.time_mesh == time) # time could be [1, 2, 3] seconds
        will return the solutions from time index [0, 4], [0, 3, 5], reverse time, and fetch specific times
        respectively.
        :param index: indices or slices of time to retrieve
        :param func: function to call on every variable in data
        :return: time filtered Mountain
        """
        return type(self)(self.mesh, self.pseudo_mesh, func(self.time_mesh[index]), self.boundaries,
                          **self.filter(index, func=func))

    def filter_space(self, index: Union[List['ellipsis'], List[int], slice],
                     func: Callable = lambda x: x) -> 'Mountain':
        """
        Filter the Mountain for a subset of space indices. For example::
            solution.filter_time([slice(0,5), 4]) # for 2D space
            solution.filter_time([0, 3, 5])
            solution.filter_time(slice(step=-1))
        will return the solutions from in space where x=[0, 5] and y=5, and x=[0, 3, 5], even reverse the first
        dimension respectively.
        :param index: indices or slices of space to retrieve
        :param func: function to call on every variable in data
        :return: space filtered Mountain
        """
        if isinstance(index, slice):
            index = [index]

        return type(self)(func(self.mesh[index]), self.pseudo_mesh, self.time_mesh, self.boundaries,
                          **self.filter([...] + index, func=func))
