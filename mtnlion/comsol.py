"""
COMSOL Data Handling.

This module is designed to load 1D solution data from a Gu & Wang reference model from COMSOL as
CSV files.
The idea is that CSV files take a long time to load, so it is more efficient to convert the data to a binary
(npz) format before processing.

COMSOL now saves it's data as a 2D matrix, however it still only uses repeated x values when the boundary solves for
different values on either side. In order to normalize the repeated bounds, all bounds are check to ensure they've got
repeated x values, such that the y values are duplicated.
"""
import logging
import os
from typing import Callable, Dict, List, Union

import numpy as np

from . import domain
from . import loader

logger = logging.getLogger(__name__)


def fix_boundaries(mesh: np.ndarray, data: np.ndarray, boundaries: Union[float, List[int], np.ndarray]) \
        -> Union[None, np.ndarray]:
    """
    Adjust COMSOL's interpretation of two-sided boundaries.

    When COMSOL outputs data from the reference model there are two solutions at every internal boundary, which causes
    COMSOL to have repeated domain values; one for the right and one for the left of the boundary. If there is only one
    internal boundary on the variable mesh at a given time, then a duplicate is added.

    :param mesh: x data to use to correct the y data
    :param data: in 2D, this would be the y data
    :param boundaries: internal boundaries
    :return: normalized boundaries to be consistent
    """
    logger.debug('Fixing boundaries: {}'.format(boundaries))
    b_indices = np.searchsorted(mesh, boundaries)

    if not len(b_indices):
        return data

    for x in b_indices[::-1]:
        if mesh[x] != mesh[x + 1]:  # add boundary
            logger.debug('Adding boundary, copying {x} at index {i}'.format(x=mesh[x], i=x))
            data = np.insert(data, x, data[x], axis=0)

    return data


def remove_dup_boundary(data: domain.ReferenceCell, item: np.ndarray) -> Union[np.ndarray, None]:
    """
    Remove points at boundaries where two values exist at the same coordinate, favor electrodes over separator.

    :param data: data in which to reference the mesh and separator indices from

    :param item: item to apply change to
    :return: Array of points with interior boundaries removed
    """
    logger.info('Removing duplicate boundaries')
    mask = np.ones(item.shape[-1], dtype=bool)
    mask[[data.sep_ind.start, data.sep_ind.stop - 1]] = False
    return item[..., mask]


def get_standardized(cell: domain.ReferenceCell) -> Union[domain.ReferenceCell, None]:
    """
    Convert COMSOL solutions to something more easily fed into FEniCS (remove repeated coordinates at boundaries).

    :param cell: reference cell to remove double boundary values from
    :return: Simplified solution cell
    """
    logger.info('Retrieving FEniCS friendly solution cell')
    return cell.filter_space(slice(0, len(cell.mesh)), func=lambda x: remove_dup_boundary(cell, x))
    # mesh = remove_dup_boundary(cell, cell.mesh)
    # new_data = {k: remove_dup_boundary(cell, v) for k, v in cell.data.items()}
    # return domain.ReferenceCell(mesh, cell.time_mesh, cell.boundaries, **new_data)


# TODO generalize the formatting of data for mesh name and arbitrary dimensions
def format_2d_data(raw_data: Dict[str, np.ndarray], boundaries: Union[float, List[int]]) \
        -> Union[Dict[str, np.ndarray], None]:
    """
    Format COMSOL stacked 1D data into a 2D matrix.

    Collect single-column 2D data from COMSOL CSV format and convert into 2D matrix for easy access, where the
    first dimension is time and the second is the solution in space. Each solution has it's own entry in a
    dictionary where the key is the name of the variable. The time step size (dt) and mesh have their own keys.

    :param raw_data: COMSOL formatted CSV files
    :param boundaries: internal boundary locations
    :return: convenient dictionary of non-stationary solutions
    """
    logger.info('Reformatting 2D data')
    data = dict()
    try:
        mesh_dict = {'time_mesh': raw_data['time_mesh'], 'mesh': raw_data['mesh'], 'boundaries': boundaries}
    except KeyError as ex:
        logger.critical('Missing required data', exc_info=True)
        raise ex

    for key, value in raw_data.items():
        if key in ('mesh', 'time_mesh', 'pseudo_mesh'):
            continue

        logger.info('Reformatting {}'.format(key))
        try:
            (x_data, y_data) = (value[:, 0], value[:, 1:])

            data[key] = fix_boundaries(x_data, y_data, boundaries).T

            if data[key].shape[-1] != len(raw_data['mesh']):
                logger.warning('{} does not fit the mesh, skipping'.format(key))
                data.pop(key, None)
            elif key not in data:
                logger.warning('{} was skipped, unknown reason'.format(key))
        except IndexError:
            logger.warning('{key} must have two columns and fit the mesh, skipping'.format(key=key), exc_info=True)
            continue
        except Exception as ex:
            logger.critical('Error occurred while formatting {key}'.format(key=key), exc_info=True)
            raise ex

        logger.info('Done formatting {}'.format(key))
    return {**data, **mesh_dict}


# TODO generalize the formatting of data for mesh name and arbitrary dimensions, also fix tools
def format_pseudo_dim(raw_data: Dict[str, np.ndarray], boundaries: Union[float, List[int]],
                      shuffle: Callable = lambda x: range(len(x))) -> Union[Dict[str, np.ndarray], None]:
    """
    Attempt to reorganize COMSOL output for the two-dimensional domain to a FEniCS friendly organization.

    :param raw_data: COMSOL output data
    :param boundaries: Domain boundaries
    :param shuffle: Function to re-organize indices
    :return: Formatted data
    """
    logger.info('Reformatting 3D data')
    data = dict()

    try:
        indices = shuffle(raw_data['pseudo_mesh'])
        mesh_dict = {'time_mesh': raw_data['time_mesh'], 'pseudo_mesh': np.array(raw_data['pseudo_mesh'])[indices],
                     'boundaries': boundaries}
    except KeyError as ex:
        logger.critical('Missing required data', exc_info=True)
        raise ex

    try:
        data['cs'] = np.array(raw_data['cs'][:, 2:])[indices].T
    except KeyError as ex:
        logger.critical('Missing required data', exc_info=True)
        raise ex

    logger.info('Done collecting {}'.format('cs'))

    return {**data, **mesh_dict}


def format_name(name: str) -> str:
    """
    Determine variable name from filename to be used in loader.collect_files.

    :param name: filename
    :return: variable name
    """
    var_name = os.path.splitext(os.path.basename(name))[0]
    if '.CSV' not in name.upper():
        logger.warning('{} does not have a CSV extension'.format(name))
    else:
        var_name = var_name.split('.', 1)[0]

    return var_name


def load(filename: str) -> domain.ReferenceCell:
    """
    Load COMSOL reference cell from formatted npz file.

    :param filename: name of the npz file
    :return: ReferenceCell
    """
    file_data = loader.load_numpy_file(filename)
    return domain.ReferenceCell.from_dict(file_data)
