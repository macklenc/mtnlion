"""
Equation solver
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import ldp


class SimMesh(object):
    """1D mesh for cell regions"""
    def __init__(self, mesh, neg, sep, pos):
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
        self.neg = neg
        self.pos = pos
        self.sep = sep


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


def parse_solution(data, time, location=None, delta_t=0.1, delete=None):
    """Fetch parameter data from a given location and time. COMSOL output is a single column that is the value of the
    function through space. When the x value resets, it indicates that a new time sample has started. Samples given in
    the tests directory are spaced at 10 samples/second.

    :param data: Imported COMSOL data
    :type data: numpy.ndarray
    :param time: Time to collect data snapshot
    :type time: int
    :param location: Location to collect data snapshot
    :type location: Union[None, None]
    :param delta_t: Change in time per frame
    :type delta_t: float
    :param delete: Delete data at given location
    :type delete: Union[None, List[int]]
    :return: Array of requested data
    :rtype: numpy.ndarray"""
    (x_data, y_data) = (data[:, 0], data[:, 1])
    # Separate time segments (frames)
    time_frame = np.nonzero(np.diff(x_data) < 0)[0]
    # Frame start indices
    start = np.insert(time_frame+1, 0, 0)
    # Frame stop indices
    stop = np.append(time_frame, len(x_data))
    # Find the samples in time for the frames
    # noinspection PyTypeChecker
    time_range = np.arange(0, len(start))*delta_t
    # Find the frame requested
    time_index = np.nonzero(time_range == time)[0][0]
    # Collect frame
    data = y_data[start[time_index]:stop[time_index]+1]
    if location:
        # Select specific value from the frame
        data = data[location == x_data[start[time_index]:stop[time_index]]]

    if delete:
        # Delete unwanted samples
        data = np.delete(data, delete)

    return np.array([data])


def nice_abs(number):
    """Return the absolute of the given number multiplied by the step function.

    :param number: Data to find absolute value
    :type number: numpy.ndarray
    :return: abs(number) if number > 0
    :rtype: numpy.ndarray
    """
    return ((np.sign(number)+1)/2)*np.abs(number)


def reaction_flux(sim_data, params, const):
    """J

    :param sim_data: Data used in calculating J
    :type sim_data: SolutionData
    :param params: Cell parameters
    :type params: Dict[str, float]
    :param const: Constants
    :type const: Dict[str, float]
    :return: Reaction flux
    :rtype: numpy.ndarray
    """

    reaction_flux0 = params['k_norm_ref'] * \
        nice_abs((params['csmax']-sim_data.cse)/params['csmax']) ** \
        (1-params['alpha']) * \
        nice_abs(sim_data.cse/params['csmax']) ** params['alpha'] * \
        nice_abs(sim_data.ce/const['ce0']) ** (1-params['alpha'])

    soc = sim_data.cse/params['csmax']
    # eta = phis-phie-params['eref'](soc)
    eta = sim_data.phis-sim_data.phie-params['Uocp'][0](soc)
    f = 96487
    r = 8.314
    return np.array([reaction_flux0*(
        np.exp((1-params['alpha'])*f*eta/(r*const['Tref'])) -
        np.exp(-params['alpha']*f*eta/(r*const['Tref'])))])


def region(mesh):
    """Find the regions in the mesh

    :param mesh: Mesh to dissect
    :type mesh: numpy.ndarray
    :return: Separated mesh
    :rtype: SimMesh
    """
    xneg = np.nonzero(mesh <= 1)[0]
    xpos = np.nonzero(mesh > 2)[0]
    xsep = np.nonzero((mesh > 1) & (mesh <= 2))[0]

    if mesh[xneg[-1]] == mesh[xneg[-2]]:
        xsep = np.concatenate((1, xneg[-1], xsep))
        xneg = np.delete(xneg, -1)

    if mesh[xsep[-1]] == mesh[xsep[-2]]:
        xpos = np.concatenate((1, xsep[-1], xpos))
        xsep = np.delete(xsep, -1)

    return SimMesh(mesh, xneg, xsep, xpos)


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


def calculate_j(time, data, params):
    jneg = np.empty((0, len(data.mesh.neg)))
    jpos = np.empty((0, len(data.mesh.pos)))

    for t in range(0, len(time)):
        jneg = np.append(jneg, reaction_flux(data.get_solution_at(t, data.mesh.neg),
                                             params['neg'], params['const']), axis=0)
        jpos = np.append(jpos, reaction_flux(data.get_solution_at(t, data.mesh.pos),
                                             params['pos'], params['const']), axis=0)

    return jneg, jpos


def rmse(estimated, true):
    return np.sqrt(((estimated - true) ** 2).mean(axis=1))


def plot_j(time, data, params, jneg, jpos):
    """

    :param time:
    :type time: List[int]
    :param data:
    :type data: SolutionData
    :param params:
    :type params: Dict[str, Dict[str, float]]
    """

    # Lneg = 100;
    # Lsep = 52;
    # Lpos = 183
    neg = data.mesh.mesh[data.mesh.neg]*params['neg']['L']
    sep = ((data.mesh.mesh[data.mesh.sep]-1)*params['sep']['L']+params['neg']['L'])
    pos = ((data.mesh.mesh[data.mesh.pos]-2)*params['pos']['L']+params['sep']['L']+params['neg']['L'])

    jsep = np.empty([1, len(sep)])[0]
    jsep[:] = np.nan

    x = np.concatenate((neg,sep,pos))*1e6
    for t in range(0, len(time)):
        j = np.concatenate((jneg[t, :], jsep, jpos[t, :]))
        # plt.plot(neg, jneg[t, :], pos, jpos[t, :])
        plt.plot(x, j)

    plt.grid()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()


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


def fetch_comsol_solutions(filename, time):
    comsol = ldp.load(filename)
    comsol_parsed = assemble_comsol(time, comsol)

    return comsol_parsed

def main():
    time = [5, 15, 25, 35, 45]
    params = fetch_params('../tests/reference/GuAndWang_parameter_list.xlsx')
    comsol_data = fetch_comsol_solutions('../tests/reference/guwang.npz', time)

    jneg, jpos = calculate_j(time, comsol_data, params)
    plot_j(time, comsol_data, params, jneg, jpos)

    rmsn = np.sum(np.abs(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.neg).j-jneg), axis=1)/len(comsol_data.mesh.neg)
    maxn = np.max(np.abs(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.neg).j), axis=1)
    rmsp = rmse(jpos, comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.pos).j)
    maxp = np.max(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.pos).j, axis=1)

    print('Neg rms: {}'.format(rmsn/maxn))
    print('Pos rms: {}'.format(rmsp/maxp))

    return


if __name__ == '__main__':
    sys.exit(main())
