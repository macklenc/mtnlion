"""
COMSOL Data Handling
"""
import logging
import os
import sys
from typing import List, Dict, Union

import click
import numpy as np


class Metadata:
    def __init__(self, data, bounds=None):
        self.data = data
        self.bounds = bounds


class ComsolData:
    """Collect and save COMSOL solutions"""

    def __init__(self, datafile: str = None, csv_file_list: List[str] = None, boundaries=None, dt: float = 0.1) -> None:
        """
        Collect data and save if both options are provided. Data is saved as a dictionary with key values being the
         basename of the file.

        :param datafile: file in which to load/save COMSOL data
        :param csv_file_list: CSV files in which to load COMSOL data
        :param dt: time between temporal samples
        """

        boundaries = [1, 2]

        logging.debug('Initializing ComsolData')
        self.datafile = datafile
        self.boundaries = boundaries
        self.data = None
        error = False

        if csv_file_list:
            logging.debug('Loading CSV files.')
            raw_params = self.collect_csv_data(csv_file_list)
            data = self.format_data(raw_params, boundaries, dt)
        elif datafile:
            logging.debug('Loading NPZ data file.')
            data = np.load(datafile)
        else:
            logging.error('Cannot do anything without CSV files or NPZ data file. Aborting.')
            return

        if csv_file_list and datafile:
            logging.info('Saving COMSOL data to NPZ: {}'.format(datafile))
            np.savez(datafile, **data)

        self.data = SolutionData(SimMesh(data['mesh']), data['ce'], data['cse'], data['phie'],
                                 data['phis'], data['j'], dt)

    def remove_dup_boundary(self, item: np.ndarray):
        # remove repeated boundary TODO: make sure this works
        mask = np.ones(item.shape[-1], dtype=bool)
        mask[[self.data.mesh.sep[0], self.data.mesh.sep[-1]]] = False
        return item[..., mask]

    @staticmethod
    def collect_csv_data(csv_file_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Collect CSV data from list of filenames and return a dictionary of the data where the key is the basename of the
         file, and the data is a 2D ndarray, where the first column is the mesh, and the second is the data. Both are
         repeated for each new time step.

        :param csv_file_list: list of files to read
        :return: extracted data
        """
        params = dict()
        logging.info('Collecting CSV data...')
        for file_name in csv_file_list:
            logging.info('Reading {}...'.format(file_name))
            # create function name from flie name
            varName, ext = os.path.splitext(os.path.basename(file_name))

            if ext.upper() != '.CSV':
                logging.warning('{} does not have a CSV extension!'.format(file_name))

            # load the data into a dictionary with the correct key name
            try:
                params[varName] = np.loadtxt(file_name, comments='%', delimiter=',', dtype=np.float64)
            except Exception as ex:
                logging.error('Failed to read {}, ignoring.  See DEBUG log.'.format(file_name))
                logging.debug(ex)

        logging.info('Finished collecting CSV data.')
        return params

    @staticmethod
    def fix_boundaries(x_data, y_data, boundaries):
        b_indices = np.searchsorted(x_data, boundaries)

        if not b_indices.any():
            return y_data

        modifier = 0
        for x in b_indices:
            if x_data[x] == x_data[x + 1]:  # swap
                y_tmp = y_data[x + modifier]
                y_data[x + modifier] = y_data[x + modifier + 1]
                y_data[x + modifier + 1] = y_tmp
            else:  # add boundary
                y_data = np.insert(y_data, x + modifier, y_data[x + modifier])
                modifier += 1

        return y_data

    @staticmethod
    def format_data(raw_params: Dict[str, np.ndarray], boundaries, dt: float) -> Union[
        Dict[str, np.ndarray], Dict[str, float]]:
        """
        Collect single-column 2D data from COMSOL CSV format and convert into 2D matrix for easy access, where the
        first dimension is time and the second is the solution in space. Each solution has it's own entry in a
        dictionary where the key is the name of the variable. The time step size (dt) and mesh have their own keys.

        :param raw_params: COMSOL formatted CSV files
        :param dt: time step for temporal problems
        :return: convenient dictionary of non-stationary solutions
        """

        logging.info('Reformatting CSV data for NPZ storage.')
        formatted_data = dict()
        formatted_data['dt'] = dt

        if 'mesh' not in raw_params:
            logging.critical('File named "mesh.csv" required.')
            raise Exception('File named "mesh.csv" required.')

        formatted_data['mesh'] = raw_params['mesh']


        # for each variable
        counter = 0
        for key, data in raw_params.items():
            if key == 'mesh':
                continue

            logging.debug('Reformatting {}.'.format(key))

            try:
                (x_data, y_data) = (data[:, 0], data[:, 1])
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


                y_data_new = np.empty([len(start), len(formatted_data['mesh'])])
                b = 0
                for i in range(len(start)):
                    b = i
                    y_section = y_data[start[i]:stop[i]]
                    x_section = x_data[start[i]:stop[i]]
                    a = ComsolData.fix_boundaries(x_section, y_section, boundaries)
                    y_data_new[i, :] = a

                # formatted_data[key] = Metadata(y_data_new, boundaries)
                formatted_data[key] = y_data_new
            except Exception as ex:
                logging.error('Error occurred while formatting {}, skipping. See DEBUG log.'.format(key))
                logging.debug(ex)

            counter += 1

        if counter == 0:
            raise Exception('No data saved')

        return formatted_data


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
        ComsolData(output, input, dt)
    except Exception as ex:
        logging.error('Unhandled exception occurred: {}'.format(ex))
        sys.exit(2)
    logging.info('Conversion completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
