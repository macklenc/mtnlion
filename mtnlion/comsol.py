"""
COMSOL Data Handling. This module is designed to load 1D solution data from a Gu & Wang reference model from COMSOL as
CSV files. The idea is that CSV files take a long time to load, so it is more efficient to convert the data to a binary
(npz) format before processing. When converting, this module will also reformat the data from COMSOL's rather confusing
output format.

My current hypothesis is that the discontinuities at the internal boundaries from the solutions are due to the boundary
being computed for both the left and right side. I.e. at x=1 on the mesh, there could be two solutions, y=1 and y=NaN.
It also appears that COMSOL for some strange reason flips the order on the boundaries, such that the first x=1 position
is for the right side of the boundary, while the second is the left side. Thus, this module will reverse these. Another
odd formatting choice from COMSOL is to conditionally add the repeated x coordinate, i.e. the first time sample of j
might have a repeated x=1, but only one x=2. In order to solve this, this module will duplicate non-repeated boundaries.
"""
import logging
import os
import sys
from typing import List, Union, Dict

import click
import numpy as np

from . import domain
from . import loader

logger = logging.getLogger(__name__)


def fix_boundaries(mesh: np.ndarray, data: np.ndarray, boundaries: Union[float, List[int], np.ndarray]) \
        -> Union[None, np.ndarray]:
    """
    When COMSOL outputs data from the reference model there are two solutions at every internal boundary, which causes
    COMSOL to have repeated domain values; one for the right and one for the left of the boundary. For some asinine
    reason, when the solution on both sides of the boundary is the same, they decided to save space and remove the
    repeated value, resulting in a mesh that fluctuates in length over time. Additionally, for some god-awful reason
    the left and right boundaries are switched in the mesh, i.e. the left boundary comes after the right boundary. This
    method is intended to fix this... If there is only one internal boundary on the variable mesh at a given time,
    then a duplicate is added, otherwise it will switch the two values.

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
        if mesh[x] == mesh[x + 1]:  # swap
            logger.debug('Swapping boundaries {x0}, {x1}, at indices {i0}, {i1}'.format(
                x0=mesh[x], x1=mesh[x + 1], i0=x, i1=x + 1))
            (data[x], data[x + 1]) = (data[x + 1], data[x])
        else:  # add boundary
            logger.debug('Adding boundary, copying {x} at index {i}'.format(x=mesh[x], i=x))
            data = np.insert(data, x, data[x])

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
    Convert COMSOL solutions to something more easily fed into FEniCS (remove repeated coordinates at boundaries)

    :param cell: reference cell to remove double boundary values from
    :return: Simplified solution cell
    """
    logger.info('Retrieving FEniCS friendly solution cell')
    return cell.filter_space(slice(0, len(cell.mesh)), func=lambda x: remove_dup_boundary(cell, x))
    # mesh = remove_dup_boundary(cell, cell.mesh)
    # new_data = {k: remove_dup_boundary(cell, v) for k, v in cell.data.items()}
    # return domain.ReferenceCell(mesh, cell.time_mesh, cell.boundaries, **new_data)


def separate_frames(mesh: np.ndarray, data: np.ndarray, boundaries: List[int]) -> np.ndarray:
    """
    COMSOL saves its data as two columns, spaces and data, which is repeated (in the same column) for every time step.
    This breaks out the 1D data in to 2D to be able to easily reference time and space. The boundaries are fixed
    so that the data has uniform length and can fit in a matrix.

    :param mesh: mesh corresponding to the data
    :param data: data that maps to the mesh
    :param boundaries: internal boundaries
    :return: 2D normalized data in time and space
    """
    logger.info('Separating data time segments')
    # Separate time segments (frames)
    time_frame = np.nonzero(np.diff(mesh) < 0)[0]
    # Frame start indices
    start = np.insert(time_frame + 1, 0, 0)
    # Frame stop indices
    stop = np.append(time_frame, len(mesh) - 1) + 1
    logger.info('Found {time} time steps'.format(time=len(start)))
    # collect y_data determining if there are discontinuous boundaries
    return np.array([fix_boundaries(mesh[start[i]:stop[i]], data[start[i]:stop[i]], boundaries)
                     for i in range(len(start))])


def format_data(raw_data: Dict[str, np.ndarray], boundaries: Union[float, List[int]])\
        -> Union[Dict[str, np.ndarray], None]:
    """
    Collect single-column 2D data from COMSOL CSV format and convert into 2D matrix for easy access, where the
    first dimension is time and the second is the solution in space. Each solution has it's own entry in a
    dictionary where the key is the name of the variable. The time step size (dt) and mesh have their own keys.

    :param raw_data: COMSOL formatted CSV files
    :param boundaries: internal boundary locations
    :return: convenient dictionary of non-stationary solutions
    """

    logger.info('Reformatting raw data')
    data = dict()
    try:
        mesh_dict = {'time_mesh': raw_data['time_mesh'], 'mesh': raw_data['mesh'], 'boundaries': boundaries}
    except KeyError as ex:
        logger.critical('Missing required data', exc_info=True)
        raise ex

    for key, value in raw_data.items():
        if key in ('mesh', 'time_mesh'):
            continue

        logger.info('Reformatting {}'.format(key))
        try:
            (x_data, y_data) = (value[:, 0], value[:, 1])
            data[key] = separate_frames(x_data, y_data, boundaries)

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


def format_name(name: str) -> str:
    """
    Determine variable name from filename to be used in loader.collect_files.

    :param name: filename
    :return: variable name
    """
    varName = os.path.splitext(os.path.basename(name))[0]
    if '.CSV' not in name.upper():
        logger.warning('{} does not have a CSV extension'.format(name))
    else:
        varName = varName.split('.', 1)[0]

    return varName


def load(filename: str) -> domain.ReferenceCell:
    """
    Load COMSOL reference cell from formatted npz file.

    :param filename: name of the npz file
    :return: ReferenceCell
    """
    file_data = loader.load_numpy_file(filename)
    return domain.ReferenceCell.from_dict(file_data)


@click.command()
@click.option('--dt', '-t', nargs=3, type=float, help='[start time stop time delta time]')
@click.option('--bound', '-b', type=click.Path(exists=True, readable=True, resolve_path=True),
              help='file containing boundaries, defaults to [1, 2]')
@click.option('--critical', 'loglevel', flag_value=logging.CRITICAL, help='Set log-level to CRITICAL')
@click.option('--error', 'loglevel', flag_value=logging.ERROR, help='Set log-level to ERROR')
@click.option('--warn', 'loglevel', flag_value=logging.WARNING, help='Set log-level to WARNING')
@click.option('--info', 'loglevel', flag_value=logging.INFO, help='Set log-level to INFO', default=True)
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG, help='Set log-level to DEBUG')
@click.argument('output', type=click.Path(writable=True, resolve_path=True))
@click.argument('input_files', nargs=-1, type=click.Path(exists=True, readable=True, resolve_path=True))
def main(input_files: List[str], output: Union[click.utils.LazyFile, str],
         dt: List[float], loglevel: Union[None, int], bound: Union[None, str]) -> Union[None, int]:
    """
    Convert COMSOL CSV files to npz.

    Create a numpy zip (npz) with variables corresponding to the csv file names.
    Each variable contains the data from the file as a list. Additionally, each
    variable is a key in the main dictionary.
    """

    logging.basicConfig(level=loglevel)

    if not input_files:
        logger.error('No CSVs were specified. Aborting.')
        sys.exit(1)

    if not bound:
        bound = [1, 2]
    else:
        bound = loader.load_csv_file(bound)

    logger.info('Output file: {}'.format(output))
    logger.info('Input file(s): {}'.format(input_files))
    logger.info('dt: {}'.format(dt))
    logger.info('boundaries: {}'.format(bound))

    file_data = loader.collect_files(input_files, format_key=format_name, loader=loader.load_csv_file)
    if 'time_mesh' not in file_data:
        try:
            file_data['time_mesh'] = np.arange(dt[0], dt[1] + dt[2], dt[2])
        except IndexError as ex:
            logger.critical('Either a dt option provided with start and stop time (inclusive) or a csv defining the '
                            'time mesh is required', exc_info=True)
            raise ex

    data = format_data(file_data, boundaries=bound)
    loader.save_npz_file(output, data)

    logger.info('Conversion completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
