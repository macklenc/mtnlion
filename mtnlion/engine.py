"""
Equation solver
"""
import logging
from typing import Union

import munch
import numpy as np

import ldp


class SimMesh(object):
    """1D mesh for reference cell. Cell regions overlap, i.e. both neg and sep contain 2."""

    def __init__(self, mesh: np.ndarray) -> None:
        """
        Store the mesh along with each individual region.

        :param mesh: Points at which to evaluate the data
        """

        logging.debug('Creating simulation mesh...')
        self.neg = self.pos = self.sep = None
        self.mesh = mesh
        self._region()

    def _unique(self, comparison):
        ind = np.nonzero(comparison)[0]

        if ind[0] > 0:
            ind = np.insert(ind, 0, ind[0]-1)

        if ind[-1] < len(self.mesh) - 1:
            ind = np.append(ind, ind[-1]+1)

        return ind

    def _region(self) -> None:
        """
        Find the reference regions in the mesh

        :param mesh: Mesh to dissect
        """

        # Find each subspace
        logging.debug('Dividing mesh into subspaces')
        self.neg = self._unique(self.mesh < 1)
        self.pos = self._unique(self.mesh > 2)
        self.sep = self._unique((self.mesh > 1) & (self.mesh < 2))


class SolutionData(object):
    """PDE Solution results for cell state"""

    def __init__(self, mesh: Union[SimMesh, float], ce: np.ndarray, cse: np.ndarray, phie: np.ndarray,
                 phis: np.ndarray, j: np.ndarray, dt: float) -> None:
        """
        Store the solutions to each cell parameter

        :param mesh: Solution mesh
        :param ce: Lithium in the electrolyte
        :param cse: Lithium between the solid and electrolyte
        :param phie: Electric potential in the electrolyte
        :param phis: Electric potential in the solid
        :param j: Rate of positive charge flowing out of a particle
        """

        logging.debug('Initializing solution data...')
        self.ce = ce
        self.cse = cse
        self.phie = phie
        self.phis = phis
        self.j = j
        self.mesh = mesh
        self.dt = dt

    def get_solution_near_time(self, time: float) -> Union[None, 'SolutionData']:
        """
        Retrieve the solution data near a given time

        :param time: time in solution to retrieve data
        :return: stationary solution
        """

        logging.debug('Retrieving solution near time: {}'.format(time))
        index = int(np.round(time / self.dt))
        logging.debug('Using time: {}'.format(index * self.dt))

        return SolutionData(self.mesh, self.ce[index], self.cse[index],
                            self.phie[index], self.phis[index], self.j[index], 0)

    def get_solution_at_time_index(self, index):
        logging.debug('Retrieving solution at time index: {}'.format(index))

        return SolutionData(self.mesh, self.ce[index, :], self.cse[index, :], self.phie[index, :],
                            self.phis[index, :], self.j[index, :], 0)

    def get_solution_near_position(self, position: float) -> 'SolutionData':
        """
        Retrieve the solution data near a given point in space

        :param position: location in solution to retrieve data
        :return: time varying solution at a given position
        """

        logging.debug('Retrieving solution near position: {}'.format(position))
        space = (np.abs(self.mesh.mesh - position)).argmin()
        logging.debug('Using position: {}'.format(space))
        return SolutionData(space, self.ce[np.newaxis, :, space], self.cse[np.newaxis, :, space],
                            self.phie[np.newaxis, :, space],
                            self.phis[np.newaxis, :, space], self.j[np.newaxis, :, space], self.dt)

    def get_solution_in_neg(self):
        logging.debug('Retrieving solution in negative electrode.')
        return SolutionData(self.mesh, self.ce[..., self.mesh.neg], self.cse[..., self.mesh.neg],
                            self.phie[..., self.mesh.neg], self.phis[..., self.mesh.neg],
                            self.j[..., self.mesh.neg], self.dt)

    def get_solution_in_pos(self):
        logging.debug('Retrieving solution in negative electrode.')
        return SolutionData(self.mesh, self.ce[..., self.mesh.pos], self.cse[..., self.mesh.pos],
                            self.phie[..., self.mesh.pos], self.phis[..., self.mesh.pos],
                            self.j[..., self.mesh.pos], self.dt)



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
