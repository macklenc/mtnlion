"""
COMSOL Data Handling
"""
import numpy as np
import sys
import ldp
import engine
import re


class SimMesh(object):
    """1D mesh for cell regions"""
    def __init__(self, mesh):
        """
        Store the entire mesh along with each individual region.

        :param mesh: Points at which to evaluate the data
        :type mesh: numpy.ndarray
        :param neg: Points in the negative electrode
        :type neg: numpy.ndarray
        :param sep: Points in the separator
        :type sep: numpy.ndarray
        :param pos: Points in the positive electrode
        :type pos: numpy.ndarray
        """
        self.mesh = mesh
        self.neg, self.pos, self.sep = region(mesh)

    def region(self, mesh):
        """Find the regions in the mesh

        :param mesh: Mesh to dissect
        :type mesh: numpy.ndarray
        :return: Separated mesh
        :rtype: SimMesh
        """
        xneg = np.nonzero(mesh <= 1)[0]
        xpos = np.nonzero(mesh >= 2)[0]
        xsep = np.nonzero((mesh >= 1) & (mesh <= 2))[0]

        if mesh[xneg[-1]] == mesh[xneg[-2]]:
            xsep = np.concatenate((1, xneg[-1], xsep))
            xneg = np.delete(xneg, -1)

        if mesh[xsep[-1]] == mesh[xsep[-2]]:
            xpos = np.concatenate((1, xsep[-1], xpos))
            xsep = np.delete(xsep, -1)

        return xneg, xsep, xpos


class ComsolData:
    def __init__(self, datafile=None, csvFileList=None, dt=0.1):
        """Collect data and save if both options are provided."""
        self.datafile = datafile
        self.data = None

        if csvFileList:
            raw_params = self.collect_csv_data(csvFileList)
            self.data = self.format_data(raw_params)
        elif datafile:
            self.data = ldp.load(datafile)

        if csvFileList and datafile:
            np.savez(datafile, **self.data)

    def collect_csv_data(self, csvFileList):
        params = dict()
        for fname in csvFileList:
            # create function name from flie name
            varName = re.sub(r'(.csv)', '', fname).split('/')[-1]

            # load the data into a dictionary with the correct key name
            params[varName] = np.loadtxt(fname, comments='%', delimiter=',', dtype=np.float64)

        return params

    def format_data(self, raw_params):
        """Collect single-column 2D from COMSOL and format into 2D matrix for easy access"""

        """Fetch parameter data from a given location and time. COMSOL output is a single column that is the value of the
        function through space. When the x value resets, it indicates that a new time sample has started. Samples given in
        the tests directory are spaced at 10 samples/second.
        """

        formatted_data = dict()
        ndata = np.empty
        for key, data in raw_params.items():
            (x_data, y_data) = (data[:, 0], data[:, 1])

            # Separate time segments (frames)
            time_frame = np.nonzero(np.diff(x_data) < 0)[0]
            # Frame start indices
            start = np.insert(time_frame+1, 0, 0)
            # Frame stop indices
            stop = np.append(time_frame, len(x_data)-1)+1
            # collect unique x_data
            formatted_data['mesh'] = np.unique(x_data[start[0]:stop[0]])
            # collect y_data removing comsol repeated number nonsense
            y_data_new = np.empty([len(start), len(formatted_data['mesh'])])

            for i in range(len(start)):
                x, ind = np.unique(x_data[start[i]:stop[i]], return_index=True)
                y_data_new[i,:] = y_data[ind]

            formatted_data[key] = y_data_new

        return formatted_data


class SolutionData(object):
    """PDE Solution results for cell state"""
    def __init__(self, mesh, ce, cse, phie, phis, j):
        """
        Store the solutions to each cell parameter

        :param mesh: Solution mesh
        :type mesh: SimMesh
        :param ce: Lithium in the electrolyte
        :type ce: numpy.ndarray
        :param cse: Lithium between the solid and electrolyte
        :type cse: numpy.ndarray
        :param phie: Electric potential in the electrolyte
        :type phie: numpy.ndarray
        :param phis: Electric potential in the solid
        :type phis: numpy.ndarray
        :param j: Rate of positive charge flowing out of a particle
        :type j: numpy.ndarray
        """
        self.ce = ce
        self.cse = cse
        self.phie = phie
        self.phis = phis
        self.j = j
        self.mesh = mesh

    def get_solution_at(self, time_index, location):
        """
        Return a SolutionData object containing solution data for one location in space and time.

        :param time_index: index in time at which to retrieve data
        :type time_index: int
        :param location: indexed location to retrieve data
        :type location: numpy.ndarray
        :return: Scalar SolutionData
        :rtype: SolutionData
        """
        return SolutionData(
            self.mesh, self.ce[time_index, location], self.cse[time_index, location],
            self.phie[time_index, location], self.phis[time_index, location],
            self.j[time_index, location])



def assemble_comsol(time, data):
    """
    Collect and parse the COMSOL data at designated times

    :param time: Times in which to collect the data frames
    :type time: List[int]
    :param data: COMSOL npz file
    :type data: numpy.lib.npyio.NpzFile
    :return: Parsed solution data
    :rtype: SolutionData
    """

    # Initialize empty arrays
    ce, cse, phie, phis, j = (np.empty((0, len(data['mesh']))) for _ in range(5))
    for ind in time:
        ce = np.append(ce, parse_solution(data['ce'], ind), axis=0)
        cse = np.append(cse, parse_solution(data['cse'], ind, delete=[80, 202]), axis=0)
        phie = np.append(phie, parse_solution(data['phie'], ind), axis=0)
        phis = np.append(phis, parse_solution(data['phis'], ind, delete=[80, 202]), axis=0)
        j = np.append(j, parse_solution(data['j'], ind, delete=[80, 202]), axis=0)

    mesh = region(data['mesh'])
    return SolutionData(mesh, ce, cse, phie, phis, j)

def fetch_params(filename):
    print('Loading Cell Parameters')
    params = dict()
    sheet = ldp.read_excel(filename, 0)
    (ncol, pcol) = (2, 3)
    params['const'] = ldp.load_params(sheet, range(7, 15), ncol, pcol)
    params['neg'] = ldp.load_params(sheet, range(18, 43), ncol, pcol)
    params['sep'] = ldp.load_params(sheet, range(47, 52), ncol, pcol)
    params['pos'] = ldp.load_params(sheet, range(55, 75), ncol, pcol)

    return params


def main():
    time = [5, 15, 25, 35, 45]
    params = fetch_params('../tests/reference/GuAndWang_parameter_list.xlsx')
    comsol_data = fetch_comsol_solutions('../tests/reference/guwang.npz', time)

    jneg, jpos = calculate_j(time, comsol_data, params)
    plot_j(time, comsol_data, jneg, jpos)
    print('Neg rms: {}'.format(rmse(jneg, comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.neg).j)))
    print('Pos rms: {}'.format(rmse(jpos, comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.pos).j)))

    return


if __name__ == '__main__':
    sys.exit(main())
