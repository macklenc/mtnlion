"""
Equation solver
"""

import munch

import ldp


def fetch_params(filename):
    print('Loading Cell Parameters')
    params = dict()
    sheet = ldp.read_excel(filename, 0)
    (ncol, pcol) = (2, 3)
    params['const'] = ldp.load_params(sheet, range(7, 15), ncol, pcol)
    params['neg'] = ldp.load_params(sheet, range(18, 43), ncol, pcol)
    params['sep'] = ldp.load_params(sheet, range(47, 52), ncol, pcol)
    params['pos'] = ldp.load_params(sheet, range(55, 75), ncol, pcol)

    return munch.DefaultMunch.fromDict(params)

# def main():
#     time = [5, 15, 25, 35, 45]
#     params = fetch_params('../tests/reference/GuAndWang_parameter_list.xlsx')
#     comsol_data = fetch_comsol_solutions('../tests/reference/guwang.npz', time)
#
#     jneg, jpos = calculate_j(time, comsol_data, params)
#     plot_j(time, comsol_data, params, jneg, jpos)
#
#     rmsn = np.sum(np.abs(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.neg).j-jneg), axis=1)/len(comsol_data.mesh.neg)
#     maxn = np.max(np.abs(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.neg).j), axis=1)
#     rmsp = rmse(jpos, comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.pos).j)
#     maxp = np.max(comsol_data.get_solution_at(slice(0, len(time)), comsol_data.mesh.pos).j, axis=1)
#
#     print('Neg rms: {}'.format(rmsn/maxn))
#     print('Pos rms: {}'.format(rmsp/maxp))
#
#     return


# if __name__ == '__main__':
#     sys.exit(main())
