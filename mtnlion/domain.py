"""Domain creation/manipulation."""
import logging
from typing import List, Tuple, Union

import numpy as np

from . import engine

logger = logging.getLogger(__name__)


def subdomain(comparison: np.ndarray) -> slice:
    """
    Find the indices of the requested subdomain, correcting for internal boundaries.

    I.e. if the mesh is defined by
    ``numpy.arange(0, 3, 0.1)`` and you wish to find the subdomain ``0 <= x <= 1`` then you would call::
        subdomain(mesh, x < 1)
    Subdomain returns ``x <= 1``, the reason for the exclusive less-than is to work around having repeated internal
    domain problems. I.e. if ``x <= 1`` had been used on a mesh with repeated boundaries at 1, then the subdomain would
    exist over both boundaries at 1.

    :param comparison: list of boolean values
    :return: indices of subdomain
    """
    start = int(np.argmax(comparison))
    stop = int(len(comparison) - np.argmax(comparison[::-1]))

    if start > 0:
        start -= 1

    if stop < len(comparison):
        stop += 1

    logger.info('Got start index: {start}, and stop index {stop}'.format(start=start, stop=stop))

    return slice(start, stop)


def subdomains(mesh: np.ndarray, regions: List[Tuple[float, float]]):
    """
    Find indices of given subdomains.

     For example separating a domain from [0, 3] into [0, 1], [1, 2], and [2, 3] would be::
        subdomains(np.arange(0, 3, 0.1), [(0, 1), (1, 2), (2, 3)])
    :param mesh: one-dimensional list of domain values
    :param regions: two dimensional list containing multiple ranges for subdomains
    :return: tuple of each subdomain indices
    """
    logger.info('Finding subdomains for regions: {}'.format(regions))
    return (subdomain((r[0] < mesh) & (mesh < r[1])) for r in regions)


class ReferenceCell(engine.Mountain):
    """
    Reference lithium-ion cell geometry, where the dimensions are normalized.

    The x dimension is defined such that the
    negative electrode exists between [0, 1], the separator exists between [1, 2], and the positive electrode exists
    between [2, 3]. For convenience the subdomains are added onto engine.Mountain.
    """

    def __init__(self, mesh: np.ndarray, pseudo_mesh: np.ndarray, time_mesh: np.ndarray,
                 boundaries: Union[np.ndarray, List[float]], **kwargs: np.ndarray) -> None:
        """
        Store the solutions to each cell parameter.

        :param mesh: Solution mesh
        :param boundaries: internal boundaries in the mesh
        :param kwargs: arrays for each solution
        """
        logger.info('Creating ReferenceCell...')
        super(ReferenceCell, self).__init__(mesh, pseudo_mesh, time_mesh, boundaries, **kwargs)

        logger.info('Finding subdomains and indices...')
        # TODO: add pseudo mesh
        self.neg_ind, self.sep_ind, self.pos_ind = subdomains(mesh[0:], [(0, 1), (1, 2), (2, 3)])
        self.neg, self.sep, self.pos = mesh[self.neg_ind, ...], mesh[self.sep_ind, ...], mesh[self.pos_ind, ...]

    def get_solution_in(self, subspace: str) -> Union[None, 'ReferenceCell']:
        """
        Return the solution for only the given subspace.

        :return: reduced solution set to only contain the given space
        """
        logging.debug('Retrieving solution in {}'.format(subspace))
        if subspace == 'neg':
            space = self.neg_ind
        elif subspace == 'sep':
            space = self.sep_ind
        elif subspace == 'pos':
            space = self.pos_ind
        else:
            return None

        return engine.Mountain(self.mesh, self.pseudo_mesh, self.time_mesh, self.boundaries, **self.data).filter_space(
            space)
