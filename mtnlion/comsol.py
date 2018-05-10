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

TODO: modify this module to support arbitrary solution data (on the same mesh?)
TODO: Get boundary info from refdomain module
"""
import logging
import os
import sys
from typing import List, Union

import click
import numpy as np


class SimMesh(object):
    """
    1D mesh for reference cell. Cell regions may overlap, i.e. both neg and sep contain x=2, and mesh will contain two
    x=2 values.

    TODO: abstract out dimensionality
    """

    def __init__(self, mesh: np.ndarray) -> None:
        """
        Store the mesh along with each individual region.

        :param mesh: Points at which to evaluate the data
        """

        logging.debug('Creating simulation mesh...')
        self.neg = self.pos = self.sep = None
        self.mesh = mesh
        self._region()

    def _unique(self, comparison: np.ndarray) -> np.ndarray:
        """
        This method will add internal repeated boundaries to subdomains if they are missing.

        :param comparison: list of boolean values
        :return: Uniform subdomain mesh
        """
        ind = np.nonzero(comparison)[0]

        if ind[0] > 0:
            ind = np.insert(ind, 0, ind[0] - 1)

        if ind[-1] < len(self.mesh) - 1:
            ind = np.append(ind, ind[-1] + 1)

        return ind

    def _region(self) -> None:
        """
        Find the reference regions in the mesh
        """

        # Find each subspace
        logging.debug('Dividing mesh into subspaces')
        self.neg = self._unique(self.mesh < 1)
        self.pos = self._unique(self.mesh > 2)
        self.sep = self._unique((self.mesh > 1) & (self.mesh < 2))


class SolutionData(object):
    """
    PDE Solution results for cell state. This class holds onto solution data imported from COMSOL, and provides
    methods to more easily interact with the data.

    TODO: abstract to work with any imported variables
    TODO: track dimensionality of data
    """

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

    def get_dict(self):
        d = self.__dict__
        d['mesh'] = self.mesh.mesh
        return d

    def get_solution_near_time(self, time: List[float]) -> Union[None, 'SolutionData']:
        """
        Retrieve the solution data near a given time

        :param time: time in solution to retrieve data
        :return: stationary solution
        """

        logging.debug('Retrieving solution near time: {}'.format(time))
        index = list(map(lambda x: int(x / self.dt), time))
        logging.debug('Using time: {}'.format(list(map(lambda x: int(x * self.dt), index))))

        return SolutionData(self.mesh, self.ce[index], self.cse[index],
                            self.phie[index], self.phis[index], self.j[index], 0)

    def get_solution_at_time_index(self, index: List) -> 'SolutionData':
        """
        Slice solution in time.

        :param index: time indices to retrieve data
        :return: time-reduced solution
        """
        logging.debug('Retrieving solution at time index: {}'.format(index))

        return SolutionData(self.mesh, self.ce[index, :], self.cse[index, :], self.phie[index, :],
                            self.phis[index, :], self.j[index, :], 0)

    def get_solution_near_position(self, position: float) -> 'SolutionData':
        """
        Retrieve the solution data near a given point in space

        :param position: location in solution to retrieve data
        :return: time varying solution at a given position

        TODO: Remove dependence on adding newaxis
        """

        logging.debug('Retrieving solution near position: {}'.format(position))
        space = (np.abs(self.mesh.mesh - position)).argmin()
        logging.debug('Using position: {}'.format(space))
        return SolutionData(space, self.ce[np.newaxis, :, space], self.cse[np.newaxis, :, space],
                            self.phie[np.newaxis, :, space],
                            self.phis[np.newaxis, :, space], self.j[np.newaxis, :, space], self.dt)

    def get_solution_in_neg(self) -> 'SolutionData':
        """
        Return the solution for only the negative electrode

        :return: reduced solution set to only contain the negative electrode space
        """
        logging.debug('Retrieving solution in negative electrode.')
        return SolutionData(self.mesh, self.ce[..., self.mesh.neg], self.cse[..., self.mesh.neg],
                            self.phie[..., self.mesh.neg], self.phis[..., self.mesh.neg],
                            self.j[..., self.mesh.neg], self.dt)

    # TODO: add get solution in sep
    def get_solution_in_pos(self) -> 'SolutionData':
        """
        Return the solution for only the positive electrode

        :return: reduced solution set to only contain the positive electrode space
        """
        logging.debug('Retrieving solution in negative electrode.')
        return SolutionData(self.mesh, self.ce[..., self.mesh.pos], self.cse[..., self.mesh.pos],
                            self.phie[..., self.mesh.pos], self.phis[..., self.mesh.pos],
                            self.j[..., self.mesh.pos], self.dt)


class IOHandler:
    """Collect COMSOL formatted CSV files/formatted npz files and save npz files."""

    def __init__(self, datafile: str = None) -> None:
        self.data = self.raw_data = None
        self.filename = datafile

        if datafile:
            logging.debug('Loading COMSOL data from npz')
            self.data = np.load(datafile)

    def collect_csv_files(self, csv_file_list: List[str] = None):
        """
        Collect CSV data from list of filenames and create a dictionary of the data where the key is the basename of the
        file, and the data is a 2D ndarray, where the first column is the mesh, and the second is the data. Both are
        repeated for each new time step.

        :param csv_file_list: list of files to read

        TODO: abstract out dimensionality requirement
        """
        params = dict()
        logging.info('Collecting CSV data...')
        for file_name in csv_file_list:
            logging.info('Reading {}...'.format(file_name))
            # create function name from file name
            varName, ext = os.path.splitext(os.path.basename(file_name))

            if ext.upper() != '.CSV':
                logging.warning('{} does not have a CSV extension!'.format(file_name))

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
        self.data = np.load(filename)


class Formatter:
    """
    Format COMSOL data to be more useful
    """

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

    def _format(self, raw_data, boundaries, dt):
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
                data[key] = np.array([
                    self._fix_boundaries(x_data[start[i]:stop[i]], y_data[start[i]:stop[i]], boundaries)
                    for i in range(len(start))])

                if data[key].shape[-1] != len(raw_data['mesh']):
                    logging.warning('{} does not fit the mesh, skipping')
                    data.pop(key, None)
                elif key not in data:
                    logging.warning('{} was skipped, unknown reason')
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

    def _fix_boundaries(self, x_data, y_data, boundaries):
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
    def set_data(data):
        return SolutionData(SimMesh(data['mesh']), data['ce'], data['cse'], data['phie'],
                            data['phis'], data['j'], data['dt'])

    @staticmethod
    def remove_dup_boundary(data: SolutionData, item: np.ndarray) -> Union[None, np.ndarray]:
        """
        Remove points at boundaries where two values exist at the same coordinate, favor electrodes over separator.
        :return: Array of points with interior boundaries removed
        """
        mask = np.ones(item.shape[-1], dtype=bool)
        mask[[data.mesh.sep[0], data.mesh.sep[-1]]] = False
        return item[..., mask]

    @staticmethod
    def get_fenics_friendly(data: SolutionData) -> SolutionData:
        """
        Convert COMSOL solutions to something more easily fed into FEniCS (remove repeated coordinates at boundaries)
        :return: Simplified solution data
        """
        mesh = Formatter.remove_dup_boundary(data, data.mesh.mesh)
        ce = Formatter.remove_dup_boundary(data, data.ce)
        cse = Formatter.remove_dup_boundary(data, data.cse)
        phie = Formatter.remove_dup_boundary(data, data.phie)
        phis = Formatter.remove_dup_boundary(data, data.phis)
        j = Formatter.remove_dup_boundary(data, data.j)

        return SolutionData(SimMesh(mesh), ce, cse, phie, phis, j, data.dt)


@click.command()
@click.option('--dt', '-t', default=0.1, type=float, help='time between samples (delta t), default=0.1')
@click.option('--critical', 'loglevel', flag_value=logging.CRITICAL, help='Set log-level to CRITICAL')
@click.option('--error', 'loglevel', flag_value=logging.ERROR, help='Set log-level to ERROR')
@click.option('--warn', 'loglevel', flag_value=logging.WARNING, help='Set log-level to WARNING')
@click.option('--info', 'loglevel', flag_value=logging.INFO, help='Set log-level to INFO', default=True)
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG, help='Set log-level to DEBUG')
@click.argument('output', type=click.Path(writable=True, resolve_path=True))
@click.argument('input', nargs=-1, type=click.Path(exists=True, readable=True, resolve_path=True))
def main(input, output, dt, loglevel):
    """
    Convert COMSOL CSV files to Mtnlion npz.

    Create a numpy zip (npz) with variables corresponding to the csv file names.
    Each variable contains the data from the file as a list. Additionally, each
    variable is a key in the main dictionary.
    """

    logging.basicConfig(level=loglevel)

    if not input:
        logging.error('No CSVs were specified. Aborting.')
        sys.exit(1)

    logging.debug('Output file: {}'.format(output))
    logging.debug('Input file(s): {}'.format(input))
    logging.debug('dt: {}'.format(dt))

    try:
        # ComsolData(output, input, dt)
        file = IOHandler()
        file.collect_csv_files(input)
        file.data = Formatter(file.raw_data, boundaries=[1, 2], dt=dt).data.get_dict()
        file.save_npz_file(output)
    except Exception as ex:
        logging.error('Unhandled exception occurred: {}'.format(ex))
        sys.exit(2)
    logging.info('Conversion completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
