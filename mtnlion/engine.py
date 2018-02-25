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
        self.mesh = mesh
        self.neg = neg
        self.pos = pos
        self.sep = sep


class SolutionData(object):
    """PDE Solution results for cell state"""
    def __init__(self, ce, cse, phie, phis, j):
        self.ce = ce
        self.cse = cse
        self.phie = phie
        self.phis = phis
        self.j = j

    def get_solution_at(self, time_index, location):
        """Return a SolutionData object containing solution data for one location in space and time"""
        return SolutionData(
                self.ce[time_index, location], self.cse[time_index, location],
                self.phie[time_index, location], self.phis[time_index, location],
                self.j[time_index, location])


def parse_solution(data, time, location=None, delta_t=0.1, delete=None):
    """Fetch parameter data from a given location and time. COMSOL output is a single column that is the value of the
    function through space. When the x value resets, it indicates that a new time sample has started. Samples given in
    the tests directory are spaced at 10 samples/second"""
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
    """Return the absolute of the given number"""
    return ((np.sign(number)+1)/2)*np.abs(number)


def reaction_flux(sim_data, params, const):
    """J"""

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
    """Find the regions in the mesh"""
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
    ce, cse, phie, phis, j = (np.empty((0, len(data['mesh']))) for _ in range(5))
    for ind in time:
        ce = np.append(ce, parse_solution(data['ce'], ind), axis=0)
        cse = np.append(cse, parse_solution(data['cse'], ind, delete=[80, 202]), axis=0)
        phie = np.append(phie, parse_solution(data['phie'], ind), axis=0)
        phis = np.append(phis, parse_solution(data['phis'], ind, delete=[80, 202]), axis=0)
        j = np.append(j, parse_solution(data['j'], ind, delete=[80, 202]), axis=0)

    return SolutionData(ce, cse, phie, phis, j)


def plot_j(time, data, mesh, params):
    jneg = np.empty((0, len(mesh.neg)))
    jpos = np.empty((0, len(mesh.pos)))

    for ind in range(0, len(time)):
        jneg = np.append(jneg, reaction_flux(data.get_solution_at(ind, mesh.neg),
                                             params['neg'], params['const']), axis=0)
        jpos = np.append(jpos, reaction_flux(data.get_solution_at(ind, mesh.pos),
                                             params['pos'], params['const']), axis=0)
        plt.plot(mesh.neg, jneg[ind, :], mesh.pos, jpos[ind, :])

    print('Neg rms: {}'.format(np.sqrt(np.mean(np.square(jneg-data.get_solution_at(
        slice(0, len(time)), mesh.neg).j), axis=1))))
    print('Pos rms: {}'.format(np.sqrt(np.mean(np.square(jpos-data.get_solution_at(
        slice(0, len(time)), mesh.pos).j), axis=1))))
    plt.grid()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()


def main():
    print('Loading Cell Parameters')
    params = dict()
    time = [5, 15, 25, 35, 45]
    sheet = ldp.read_excel(
        '../tests/reference/GuAndWang_parameter_list.xlsx', 0)
    (ncol, pcol) = (2, 3)
    params['const'] = ldp.load_params(sheet, range(7, 15), ncol, pcol)
    params['neg'] = ldp.load_params(sheet, range(18, 43), ncol, pcol)
    params['sep'] = ldp.load_params(sheet, range(47, 52), ncol, pcol)
    params['pos'] = ldp.load_params(sheet, range(55, 75), ncol, pcol)

    comsol = ldp.load('../tests/reference/guwang.npz')

    comsol_parsed = assemble_comsol(time, comsol)

    comsol_mesh = region(comsol['mesh'])
    # print(reaction_flux(comsol_parsed, params['pos'], params['const']))
    plot_j(time, comsol_parsed, comsol_mesh, params)

    return


if __name__ == '__main__':
    sys.exit(main())
