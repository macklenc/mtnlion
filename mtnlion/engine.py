"""
Equation solver
"""
import logging
from typing import Union, Dict, List

import munch
import numpy as np

import ldp

logger = logging.getLogger(__name__)


def rmse(estimated, true):
    return np.sqrt(((estimated - true) ** 2).mean(axis=1))


def fetch_params(filename):
    print('Loading Cell Parameters')
    params = dict()
    sheet = ldp.read_excel(filename, 0)
    (ncol, pcol) = (2, 3)
    params['const'] = ldp.load_params(sheet, range(7, 15), ncol, pcol)
    params['neg'] = ldp.load_params(sheet, range(18, 43), ncol, pcol)
    params['sep'] = ldp.load_params(sheet, range(47, 52), ncol, pcol)
    params['pos'] = ldp.load_params(sheet, range(55, 75), ncol, pcol)

    return munch.DefaultMunch.fromDict(params)


def find_ind(data, value):
    return np.nonzero(np.in1d(data, value))


def _dummy(item):
    """Do nothing"""
    return item


class Mountain:
    """
    Container for holding n-variable n-dimensional data in space and time.
    """

    def __init__(self, mesh: Union[np.ndarray, float], time_mesh: Union[np.ndarray, float],
                 boundaries: np.ndarray, **kwargs) -> None:
        """
        Store the solutions to each given parameter

        :param mesh: Solution mesh
        :param boundaries: internal boundaries in the mesh
        :param kwargs: arrays for each solution
        """

        logger.info('Initializing solution data...')
        self.data = munch.Munch(kwargs)
        self.mesh = mesh
        self.time_mesh = time_mesh
        self.boundaries = boundaries

    def to_dict(self) -> Union[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Retrieve dictionary of Mountain to serialize
        :return: data dictionary
        """
        d = {'mesh': self.mesh, 'time_mesh': self.time_mesh, 'boundaries': self.boundaries}
        return dict(d, **self.data)

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Convert dictionary to SolutionData
        :param data: dictionary of formatted data
        :return: consolidated simulation data
        """
        return cls(data.pop('mesh'), data.pop('time_mesh'), data.pop('boundaries'), **data)

    def filter(self, index: Union[List['ellipsis'], List[int], List[slice], slice], func=None) -> Dict[str, np.ndarray]:
        """
        Filter through dictionary to collect sections of the contained ndarrays.
        :param index: subset of arrays to collect
        :return: dictionary of reduced arrays
        """
        if func is None:
            func = _dummy

        return {k: func(v[index]) for k, v in self.data.items() if not np.isscalar(v)}

    def filter_time(self, index: Union[List['ellipsis'], List[int], slice], func=None) -> Union[None, 'Mountain']:
        """
        Filter the Mountain for a subset of time indices. For example::
            solution.filter_time(slice(0,5))
            solution.filter_time([0, 3, 5])
            solution.filter_time(slice(step=-1))
            solution.filter_time(numpy.where(solution.time_mesh == time) # time could be [1, 2, 3] seconds
        will return the solutions from time index [0, 4], [0, 3, 5], reverse time, and fetch specific times
        respectively.
        :param index: indices or slices of time to retrieve
        :return: time filtered Mountain
        """
        if func is None:
            func = _dummy

        return type(self)(self.mesh, func(self.time_mesh[index]), self.boundaries, **self.filter(index, func=func))

    def filter_space(self, index: Union[List['ellipsis'], List[int], slice], func=None) -> Union[None, 'Mountain']:
        """
        Filter the Mountain for a subset of space indices. For example::
            solution.filter_time([slice(0,5), 4]) # for 2D space
            solution.filter_time([0, 3, 5])
            solution.filter_time(slice(step=-1))
        will return the solutions from in space where x=[0, 5] and y=5, and x=[0, 3, 5], even reverse the first
        dimension respectively.
        :param index: indices or slices of space to retrieve
        :return: space filtered Mountain
        """

        if isinstance(index, slice):
            index = [index]

        if func is None:
            func = _dummy

        return type(self)(func(self.mesh[index]), self.time_mesh, self.boundaries,
                          **self.filter([...] + index, func=func))


# def main():
#     time = [5, 15, 25, 35, 45]
#     params = fetch_params('../tests/reference/GuAndWang_parameter_list.xlsx')
#     comsol_data = fetch_comsol_solutions('../tests/reference/guwang.npz', time)
#
#     jneg, jpos = calculate_j(time, comsol_data, params)
#     plot_j(time, comsol_data, params, jneg, jpos)
#
#     rmsn = np.sum(np.abs(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.neg).j-jneg), axis=1)/len(comsol_data.mesh.neg)
#     maxn = np.max(np.abs(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.neg).j), axis=1)
#     rmsp = rmse(jpos, comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.pos).j)
#     maxp = np.max(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.pos).j, axis=1)
#
#     print('Neg rms: {}'.format(rmsn/maxn))
#     print('Pos rms: {}'.format(rmsp/maxp))
#
#     return


# if __name__ == '__main__':
#     sys.exit(main())
