import sys

import matplotlib.pyplot as plt
import numpy as np

import mtnlion.comsol as comsol
import mtnlion.engine as engine


def nice_abs(number):
    """Return the absolute of the given number multiplied by the step function.

    :param number: Data to find absolute value
    :type number: numpy.ndarray
    :return: abs(number) if number > 0
    :rtype: numpy.ndarray
    """
    return ((np.sign(number) + 1) / 2) * np.abs(number)


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

    reaction_flux0 = params.k_norm_ref * \
                     nice_abs((params.csmax - sim_data.cse) / params.csmax) ** \
                     (1 - params.alpha) * \
                     nice_abs(sim_data.cse / params.csmax) ** params.alpha * \
                     nice_abs(sim_data.ce / const.ce0) ** (1 - params.alpha)

    soc = sim_data.cse / params.csmax
    # eta = phis-phie-params['eref'](soc)
    eta = sim_data.phis - sim_data.phie - params.Uocp[0](soc)
    f = 96487
    r = 8.314
    j = reaction_flux0 * (
        np.exp((1 - params.alpha) * f * eta / (r * const.Tref)) -
        np.exp(-params.alpha * f * eta / (r * const.Tref)))

    return j


def calculate_j(data, params):
    negdata = data.get_solution_in('neg')
    posdata = data.get_solution_in('pos')

    jneg = reaction_flux(negdata.data, params.neg, params.const)
    jpos = reaction_flux(posdata.data, params.pos, params.const)

    return jneg, jpos


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
    neg = data.neg * params['neg']['L']
    sep = ((data.sep - 1) * params['sep']['L'] + params['neg']['L'])
    pos = ((data.pos - 2) * params['pos']['L'] + params['sep']['L'] + params['neg']['L'])

    jsep = np.empty([1, len(sep)])[0]
    jsep[:] = np.nan

    x = np.concatenate((neg, sep, pos)) * 1e6
    for t in range(0, len(time)):
        j = np.concatenate((jneg[t], jsep, jpos[t]))
        # plt.plot(neg, jneg[t, :], pos, jpos[t, :])
        plt.plot(x, j)

    plt.grid()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()


def main():
    import timeit
    time = [0, 5, 10, 15, 20]
    resources = '../reference/'
    params = engine.fetch_params(resources + 'GuAndWang_parameter_list.xlsx')
    d_comsol = comsol.load(resources + 'guwang.npz')
    time_ind = engine.find_ind(d_comsol.time_mesh, time)
    d_comsol = d_comsol.filter_time(time_ind)
    # d_comsol = comsol.ComsolData(resources + 'guwang.npz')

    st = timeit.default_timer()
    jneg, jpos = calculate_j(d_comsol, params)
    plot_j(time, d_comsol, params, jneg, jpos)
    print('Time: {}'.format(timeit.default_timer()-st))
    print(engine.rmse(jneg, d_comsol.data.j[:, d_comsol.neg_ind]))
    print(engine.rmse(jpos, d_comsol.data.j[:, d_comsol.pos_ind]))

    # jneg_orig = d_comsol.data.get_solution_in_neg().get_solution_at_time_index(list(map(lambda x: x*10, time))).j
    # jpos_orig = d_comsol.data.get_solution_in_pos().get_solution_at_time_index(list(map(lambda x: x*10, time))).j
    # rmsn = np.sum(np.abs(jneg_orig - jneg), axis=1) / len(d_comsol.data.mesh.neg)
    # maxn = np.max(np.abs(jneg_orig), axis=1)
    # rmsp = rmse(jpos, jpos_orig)
    # maxp = np.max(jpos_orig, axis=1)

    # print('Neg rms: {}'.format(np.log10(rmsn / maxn)))
    # print('Pos rms: {}'.format(np.log10(rmsp / maxp)))

    return


if __name__ == '__main__':
    sys.exit(main())
