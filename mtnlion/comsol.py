"""
COMSOL Data Handling
"""
import numpy as np
import sys
import os
import argparse
import logging
from typing import List, Dict, Union

import ldp
# import engine


class SimMesh(object):
    """1D mesh for reference cell. Cell regions overlap, i.e. both neg and sep contain 2."""

    def __init__(self, mesh: np.ndarray) -> None:
        """
        Store the mesh along with each individual region.

        :param mesh: Points at which to evaluate the data
        """

        logging.debug('Creating simulation mesh...')
        self.neg, self.pos, self.sep = None, None, None
        self.mesh = mesh
        self.region(mesh)

    def region(self, mesh: np.ndarray) -> None:
        """
        Find the reference regions in the mesh

        :param mesh: Mesh to dissect
        """

        # Find each subspace
        logging.debug('Dividing mesh into subspaces')
        xneg = np.nonzero(mesh <= 1)[0]
        xpos = np.nonzero(mesh >= 2)[0]
        xsep = np.nonzero((mesh >= 1) & (mesh <= 2))[0]

        # Remove COMSOL repeated values if necessary
        if mesh[xneg[-1]] == mesh[xneg[-2]]:
            logging.debug('Found repeated value in negative electrode: {}, correcting.'.format(xneg[-1]))
            xsep = np.concatenate((1, xneg[-1], xsep))
            xneg = np.delete(xneg, -1)

        # Remove COMSOL repeated values if necessary
        if mesh[xsep[-1]] == mesh[xsep[-2]]:
            logging.debug('Found repeated value in separator: {}, correcting.'.format(xneg[-1]))
            xpos = np.concatenate((1, xsep[-1], xpos))
            xsep = np.delete(xsep, -1)

        self.neg, self.pos, self.sep = xneg, xsep, xpos


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

    def get_solution_at_time(self, time: float) -> Union[None, 'SolutionData']:
        """
        Retrieve the solution data near a given time

        :param time: time in solution to retrieve data
        :return: stationary solution
        """

        logging.debug('Retrieving solution near time: {}'.format(time))
        index = int(np.round(time/self.dt))
        logging.debug('Using time: {}'.format(index*self.dt))
        return SolutionData(self.mesh, self.ce[index, :], self.cse[index, :], self.phie[index, :],
                            self.phis[index, :], self.j[index, :], 0)

    def get_solution_in_space(self, position: float) -> 'SolutionData':
        """
        Retrieve the solution data near a given point in space

        :param position: location in solution to retrieve data
        :return: time varying solution at a given position
        """

        logging.debug('Retrieving solution near position: {}'.format(position))
        space = (np.abs(self.mesh.mesh - position)).argmin()
        logging.debug('Using position: {}'.format(space))
        return SolutionData(space, self.ce[:, space], self.cse[:, space], self.phie[:, space],
                            self.phis[:, space], self.j[:, space], self.dt)


class ComsolData:
    """Collect and save COMSOL solutions"""
    def __init__(self, datafile: str = None, csv_file_list: List[str] = None, dt: float = 0.1) -> None:
        """
        Collect data and save if both options are provided. Data is saved as a dictionary with key values being the
         basename of the file.

        :param datafile: file in which to load/save COMSOL data
        :param csv_file_list: CSV files in which to load COMSOL data
        :param dt: time between temporal samples
        """

        logging.debug('Initializing ComsolData')
        self.datafile = datafile
        self.data = None

        if csv_file_list:
            logging.debug('Loading CSV files.')
            raw_params = self.collect_csv_data(csv_file_list)
            data = self.format_data(raw_params, dt)
        elif datafile:
            logging.debug('Loading NPZ data file.')
            data = ldp.load(datafile)
        else:
            logging.error('Cannot do anything without CSV files or NPZ data file. Aborting.')
            return

        if csv_file_list and datafile:
            logging.info('Saving COMSOL data to NPZ: {}'.format(datafile))
            np.savez(datafile, **data)

        mesh = SimMesh(data['mesh'])
        self.data = SolutionData(mesh, data['ce'], data['cse'], data['phie'], data['phis'], data['j'], dt)

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
                logging.warning('Failed to read {}, ignoring.  See DEBUG log.'.format(file_name))
                logging.debug(ex)

        logging.info('Finished collecting CSV data.')
        return params

    @staticmethod
    def format_data(raw_params: Dict[str, np.ndarray], dt: float) -> Union[Dict[str, np.ndarray], Dict[str, float]]:
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

        # for each variable
        for key, data in raw_params.items():
            logging.debug('Reformatting {}.'.format(key))

            try:
                (x_data, y_data) = (data[:, 0], data[:, 1])
            except Exception as ex:
                logging.warning('{} must have two columns, skipping. See DEBUG log.'.format(key))
                logging.debug(ex)
                break

            try:
                # Separate time segments (frames)
                time_frame = np.nonzero(np.diff(x_data) < 0)[0]
                # Frame start indices
                start = np.insert(time_frame + 1, 0, 0)
                # Frame stop indices
                stop = np.append(time_frame, len(x_data) - 1) + 1
                # collect unique x_data
                if 'mesh' not in formatted_data:
                    formatted_data['mesh'] = np.unique(x_data[start[0]:stop[0]])
                # collect y_data removing comsol repeated number nonsense
                y_data_new = np.empty([len(start), len(formatted_data['mesh'])])

                # Remove y data for repeated mesh values
                for i in range(len(start)):
                    x, ind = np.unique(x_data[start[i]:stop[i]], return_index=True)
                    y_data_new[i, :] = y_data[ind]

                formatted_data[key] = y_data_new
            except Exception as ex:
                logging.warning('Error occurred while formatting {}, skipping. See DEBUG log.'.format(key))
                logging.debug(ex)

        return formatted_data


def main():
    """
    Create a numpy zip (npz) with variables corresponding to the csv file names.
    Each variable contains the data from the file as a list. Additionally, each
    variable is a key in the main dictionary.
    """

    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Convert COMSOL CSV files to Mtnlion npz.')
    parser.add_argument('output', metavar='output', type=str, nargs=1, help='output npz file')
    parser.add_argument('inputs', metavar='inputs', nargs='+', help='input CSV files')
    parser.add_argument('-t', '--dt', type=float, nargs=1, help='time between samples (delta t), default=0.1',
                        default=0.1)

    args = parser.parse_args()

    logging.debug('Output file: {}'.format(args.output[0]))
    logging.debug('Input file(s): {}'.format(args.inputs))
    logging.debug('dt: {}'.format(args.dt))

    ComsolData(args.output[0], args.inputs, args.dt)
    logging.info('Conversion completed successfully')
    return


if __name__ == '__main__':
    sys.exit(main())
