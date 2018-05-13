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

import domain
import loader

logger = logging.getLogger(__name__)


class Formatter:
    """
    Format COMSOL data to be more useful
    """
    data: 'domain.ReferenceCell'

    def __init__(self, raw_data, boundaries=None, dt: float = 0.1) -> None:
        """
        Format a "raw" data dictionary, where each data element is assumed to be in COMSOL's asinine format, with the
        exception of 'mesh' which must exist in the dictionary. If boundaries is provided, an attempt will be made to
        deal with duplicate internal boundary data. The data is also assumed to be formatted such that it has two
        columns, the first is the mesh that the data is on, and the second is the data. The data is stacked in time,
        so every time the mesh restarts, that data will be saved as a new time step. Each time step is separated by dt
        time.

        :param raw_data: COMSOL formatted data dictionary
        :param boundaries: internal boundaries that may need correction
        :param dt: time change per step in data
        """
        self.data = None
        self._format(raw_data, boundaries, dt)

    def _format(self, raw_data: Dict[str, np.ndarray], boundaries: List[int], dt: float) -> None:
        """
        Collect single-column 2D data from COMSOL CSV format and convert into 2D matrix for easy access, where the
        first dimension is time and the second is the solution in space. Each solution has it's own entry in a
        dictionary where the key is the name of the variable. The time step size (dt) and mesh have their own keys.

        :param raw_data: COMSOL formatted CSV files
        :param boundaries: internal boundary locations
        :param dt: change in time between each sample
        :return: convenient dictionary of non-stationary solutions
        """

        logging.info('Reformatting raw data')
        data = dict()

        if 'mesh' not in raw_data:
            logging.critical('Data named "mesh" required')
            raise Exception('Data named "mesh" required')

        # for each variable
        for key, value in raw_data.items():
            if key == 'mesh':
                continue

            logging.debug('Reformatting {}.'.format(key))

            try:
                (x_data, y_data) = (value[:, 0], value[:, 1])
            except Exception as ex:
                logging.warning('{} must have two columns, skipping. See DEBUG log.'.format(key))
                logging.debug(ex)
                continue

            try:
                # Separate time segments (frames)
                time_frame = np.nonzero(np.diff(x_data) < 0)[0]
                # Frame start indices
                start = np.insert(time_frame + 1, 0, 0)
                # Frame stop indices
                stop = np.append(time_frame, len(x_data) - 1) + 1
                # collect y_data determining if there are discontinuous boundaries
                logging.debug('Separating data time segments and fixing boundaries')
                data[key] = np.array([
                    Formatter._fix_boundaries(x_data[start[i]:stop[i]], y_data[start[i]:stop[i]], boundaries)
                    for i in range(len(start))])

                if data[key].shape[-1] != len(raw_data['mesh']):
                    logging.warning('{} does not fit the mesh, skipping'.format(key))
                    data.pop(key, None)
                elif key not in data:
                    logging.warning('{} was skipped, unknown reason'.format(key))
            except Exception as ex:
                logging.error('Error occurred while formatting {}, skipping. See DEBUG log.'.format(key))
                logging.debug(ex)

        if not len(data):
            logging.critical('No data recovered, aborting')
            raise Exception('No data saved')

        data['time_mesh'] = dt
        data['mesh'] = raw_data['mesh']
        data['boundaries'] = boundaries

        self.data = self.set_data(data)

    @staticmethod
    def _fix_boundaries(x_data: np.ndarray, y_data: np.ndarray, boundaries: List[int]) -> np.ndarray:
        b_indices = np.searchsorted(x_data, boundaries)

        if not len(b_indices):
            return y_data

        for x in b_indices[::-1]:
            if x_data[x] == x_data[x + 1]:  # swap
                (y_data[x], y_data[x + 1]) = (y_data[x + 1], y_data[x])
            else:  # add boundary
                y_data = np.insert(y_data, x, y_data[x])

        return y_data

    @staticmethod
    def set_data(data: Dict):
        """
        Convert dictionary to SolutionData
        :param data: dictionary of formatted data
        :return: consolidated simulation data
        """
        return domain.ReferenceCell(data.pop('mesh'), data.pop('time_mesh'), data.pop('boundaries'), **data)

    @staticmethod
    def remove_dup_boundary(data: 'domain.ReferenceCell', item: np.ndarray) -> Union[None, np.ndarray]:
        """
        Remove points at boundaries where two values exist at the same coordinate, favor electrodes over separator.
        :return: Array of points with interior boundaries removed
        """
        logging.debug('Removing duplicate boundaries')
        mask = np.ones(item.shape[-1], dtype=bool)
        mask[[data.sep_ind.start, data.sep_ind.stop - 1]] = False
        return item[..., mask]

    @staticmethod
    def get_fenics_friendly(data: 'domain.ReferenceCell') -> 'domain.ReferenceCell':
        """
        Convert COMSOL solutions to something more easily fed into FEniCS (remove repeated coordinates at boundaries)
        :return: Simplified solution data
        """
        logging.debug('Retrieving FEniCS friendly solution data')
        mesh = Formatter.remove_dup_boundary(data, data.mesh)
        new_data = {k: Formatter.remove_dup_boundary(data, v) for k, v in data.data.items()}

        return domain.ReferenceCell(mesh, data.time_mesh, data.boundaries, **new_data)


def format_name(name):
    varName = os.path.splitext(os.path.basename(name))[0]
    if '.CSV' not in name.upper():
        logging.warning('{} does not have a CSV extension!'.format(name))
    else:
        varName = varName.split('.', 1)[0]

    return varName


def load(filename):
    file_data = loader.load_numpy_file(filename)
    return Formatter.set_data(file_data)

@click.command()
@click.option('--dt', '-t', default=0.1, type=float, help='time between samples (delta t), default=0.1')
@click.option('--critical', 'loglevel', flag_value=logging.CRITICAL, help='Set log-level to CRITICAL')
@click.option('--error', 'loglevel', flag_value=logging.ERROR, help='Set log-level to ERROR')
@click.option('--warn', 'loglevel', flag_value=logging.WARNING, help='Set log-level to WARNING')
@click.option('--info', 'loglevel', flag_value=logging.INFO, help='Set log-level to INFO', default=True)
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG, help='Set log-level to DEBUG')
@click.argument('output', type=click.Path(writable=True, resolve_path=True))
@click.argument('input_files', nargs=-1, type=click.Path(exists=True, readable=True, resolve_path=True))
# TODO: allow importing time mesh -- fix dt
def main(input_files: List[str], output: Union[click.utils.LazyFile, str],
         dt: float, loglevel: Union[None, int]) -> Union[None, int]:
    """
    Convert COMSOL CSV files to npz.

    Create a numpy zip (npz) with variables corresponding to the csv file names.
    Each variable contains the data from the file as a list. Additionally, each
    variable is a key in the main dictionary.
    """

    logging.basicConfig(level=loglevel)

    if not input_files:
        logging.error('No CSVs were specified. Aborting.')
        sys.exit(1)

    logging.debug('Output file: {}'.format(output))
    logging.debug('Input file(s): {}'.format(input_files))
    logging.debug('dt: {}'.format(dt))

    dt = np.arange(0, 50.1, dt)

    try:
        file_data = loader.collect_files(input_files, format_key=format_name, loader=loader.load_csv_file)
        data = Formatter(file_data, boundaries=[1, 2], dt=dt).data.get_dict()
        loader.save_npz_file(output, data)
    except Exception as ex:
        logging.error('Unhandled exception occurred: {}'.format(ex))
        raise ex
    logging.info('Conversion completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
