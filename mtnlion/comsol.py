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
from typing import List, Union, Dict, Tuple

import click
import munch
import numpy as np


def subdomain(comparison: np.ndarray) -> slice:
    """
    Find the indices of the requested subdomain, correcting for internal boundaries. I.e. if the mesh is defined by
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

    return slice(start, stop)


def subdomains(mesh: np.ndarray, regions: List[Tuple[float, float]]):
    """
    Find indices of given subdomains. For example separating a domain from [0, 3] into [0, 1], [1, 2], and [2, 3] would
    be::
        subdomains(np.arange(0, 3, 0.1), [(0, 1), (1, 2), (2, 3)])
    :param mesh: one-dimensional list of domain values
    :param regions: two dimensional list containing multiple ranges for subdomains
    :return: tuple of each subdomain indices
    """
    return (subdomain((r[0] < mesh) & (mesh < r[1])) for r in regions)


class SolutionData(object):
    """
    PDE Solution results for cell state. This class holds onto solution data imported from COMSOL, and provides
    methods to more easily interact with the data.
    """

    def __init__(self, mesh: Union[np.ndarray, float], dt: float, boundaries: np.ndarray, **kwargs) -> None:
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
        self.data = munch.Munch(kwargs)
        self.mesh = mesh
        self.neg_ind, self.sep_ind, self.pos_ind = subdomains(mesh[0:], [(0, 1), (1, 2), (2, 3)])
        self.neg, self.sep, self.pos = mesh[self.neg_ind, ...], mesh[self.sep_ind, ...], mesh[self.pos_ind, ...]
        self.dt = dt
        self.boundaries = boundaries

    def get_dict(self) -> Union[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Retrieve dictionary of SolutionData to serialize
        :return: data dictionary
        """
        d = {'mesh': self.mesh, 'dt': self.dt, 'boundaries': self.boundaries}
        return dict(d, **self.data)

    def _filter(self, index: Union[List['ellipsis'], List[int], slice]) -> Dict[str, np.ndarray]:
        """
        Filter through dictionary to collect sections of the contained ndarrays.
        :param index: subset of arrays to collect
        :return: dictionary of reduced arrays
        """
        return {k: v[index] for k, v in self.data.items() if not np.isscalar(v)}

    def get_solution_near_time(self, time: List[float]) -> Union[None, 'SolutionData']:
        """
        Retrieve the solution data near a given time

        :param time: time in solution to retrieve data
        :return: stationary solution
        """

        logging.debug('Retrieving solution near time: {}'.format(time))
        index = list(map(lambda x: int(x / self.dt), time))
        logging.debug('Using time: {}'.format(list(map(lambda x: int(x * self.dt), index))))

        return SolutionData(self.mesh, 0, self.boundaries, **self._filter(index))

    def get_solution_at_time_index(self, index: List) -> 'SolutionData':
        """
        Slice solution in time.

        :param index: time indices to retrieve data
        :return: time-reduced solution
        """
        logging.debug('Retrieving solution at time index: {}'.format(index))

        return SolutionData(self.mesh, 0, self.boundaries, **self._filter([index, ...]))

    def get_solution_near_position(self, position: float) -> 'SolutionData':
        """
        Retrieve the solution data near a given point in space

        :param position: location in solution to retrieve data
        :return: time varying solution at a given position
        """

        logging.debug('Retrieving solution near position: {}'.format(position))
        space = (np.abs(self.mesh - position)).argmin()
        logging.debug('Using position: {}'.format(space))
        return SolutionData(space, self.dt, self.boundaries, **self._filter([..., space]))

    def get_solution_in(self, subspace: str) -> Union[None, 'SolutionData']:
        """
        Return the solution for only the given subspace

        :return: reduced solution set to only contain the given space
        """
        logging.debug('Retrieving solution in {}'.format(subspace))
        if subspace is 'neg':
            space = self.neg_ind
        elif subspace is 'sep':
            space = self.sep_ind
        elif subspace is 'pos':
            space = self.pos_ind
        else:
            return None

        return SolutionData(self.mesh, self.dt, self.boundaries, **self._filter([..., space]))


class IOHandler:
    """Collect COMSOL formatted CSV files/formatted npz files and save npz files."""

    def __init__(self, datafile: str = None) -> None:
        self.data = self.raw_data = None
        self.filename = datafile

        if datafile:
            logging.debug('Loading COMSOL data from npz')
            self.load_npz_file()

    def collect_csv_files(self, csv_file_list: List[str] = None):
        """
        Collect CSV data from list of filenames and create a dictionary of the data where the key is the basename of the
        file, and the data is a 2D ndarray, where the first column is the mesh, and the second is the data. Both are
        repeated for each new time step. Cannot read entire file names if they contain extra periods that do not proceed
        an extension. I.e. j.csv.bz2 or j.csv are okay, but my.file.csv is not.

        :param csv_file_list: list of files to read

        TODO: abstract out dimensionality requirement
        """
        params = dict()
        logging.info('Collecting CSV data...')
        for file_name in csv_file_list:
            logging.info('Reading {}...'.format(file_name))
            # create function name from file name
            varName = os.path.splitext(os.path.basename(file_name))[0]

            if '.CSV' not in file_name.upper():
                logging.warning('{} does not have a CSV extension!'.format(file_name))
            else:
                varName = varName.split('.', 1)[0]

            # load the data into a dictionary with the correct key name
            try:
                params[varName] = np.loadtxt(file_name, comments='%', delimiter=',', dtype=np.float64)
            except Exception as ex:
                logging.error('Failed to read {}, ignoring.  See DEBUG log.'.format(file_name))
                logging.debug(ex)

        self.raw_data = params
        logging.info('Finished collecting CSV data.')

    def save_npz_file(self, filename: str = None):
        """
        Save self.data to an npz file. If no filename is provided, the filename that was used to load the data will be
        used.

        :param filename: Name of the npz file
        """
        logging.info('Saving data to npz: {}'.format(filename))
        if not filename:
            filename = self.filename
        np.savez(filename, **self.data)

    def load_npz_file(self, filename: str = None):
        """
        Load self.data from an npz file. If no filename is provided, the filename that was used to load the data will be
        used.

        :param filename: Name of the npz file
        """
        logging.info('Loading data from npz: {}'.format(filename))
        if not filename:
            filename = self.filename
        else:
            self.filename = filename

        with np.load(filename) as data:
            self.data = {k: v for k, v in data.items()}


class Formatter:
    """
    Format COMSOL data to be more useful
    """
    data: 'SolutionData'

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

        data['dt'] = dt
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
        return SolutionData(data.pop('mesh'), data.pop('dt'), data.pop('boundaries'), **data)

    @staticmethod
    def remove_dup_boundary(data: SolutionData, item: np.ndarray) -> Union[None, np.ndarray]:
        """
        Remove points at boundaries where two values exist at the same coordinate, favor electrodes over separator.
        :return: Array of points with interior boundaries removed
        """
        logging.debug('Removing duplicate boundaries')
        mask = np.ones(item.shape[-1], dtype=bool)
        mask[[data.sep_ind.start, data.sep_ind.stop - 1]] = False
        return item[..., mask]

    @staticmethod
    def get_fenics_friendly(data: SolutionData) -> SolutionData:
        """
        Convert COMSOL solutions to something more easily fed into FEniCS (remove repeated coordinates at boundaries)
        :return: Simplified solution data
        """
        logging.debug('Retrieving FEniCS friendly solution data')
        mesh = Formatter.remove_dup_boundary(data, data.mesh)
        new_data = {k: Formatter.remove_dup_boundary(data, v) for k, v in data.data.items()}

        return SolutionData(mesh, data.dt, data.boundaries, **new_data)


@click.command()
@click.option('--dt', '-t', default=0.1, type=float, help='time between samples (delta t), default=0.1')
@click.option('--critical', 'loglevel', flag_value=logging.CRITICAL, help='Set log-level to CRITICAL')
@click.option('--error', 'loglevel', flag_value=logging.ERROR, help='Set log-level to ERROR')
@click.option('--warn', 'loglevel', flag_value=logging.WARNING, help='Set log-level to WARNING')
@click.option('--info', 'loglevel', flag_value=logging.INFO, help='Set log-level to INFO', default=True)
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG, help='Set log-level to DEBUG')
@click.argument('output', type=click.Path(writable=True, resolve_path=True))
@click.argument('input_files', nargs=-1, type=click.Path(exists=True, readable=True, resolve_path=True))
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

    try:
        # ComsolData(output, input_files, dt)
        file = IOHandler()
        file.collect_csv_files(input_files)
        file.data = Formatter(file.raw_data, boundaries=[1, 2], dt=dt).data.get_dict()
        file.save_npz_file(output)
    except Exception as ex:
        logging.error('Unhandled exception occurred: {}'.format(ex))
        sys.exit(2)
    logging.info('Conversion completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
