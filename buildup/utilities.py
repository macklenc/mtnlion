import os
import time

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import mtnlion.comsol as comsol
import mtnlion.engine as engine


def gather_data():
    # Load required cell data
    localdir = os.path.dirname(__file__)
    resources = os.path.join(localdir, 'reference/')
    params = engine.fetch_params(os.path.join(resources, 'GuAndWang_parameter_list.xlsx'))
    d_comsol = comsol.load(os.path.join(resources, 'guwang.npz'))
    return d_comsol, params


def create_solution_matrices(nr, nc, r):
    return tuple(np.empty((nr, nc)) for _ in range(r))


def create_functions(V, r):
    return tuple(fem.Function(V) for _ in range(r))


def assign_functions(from_funcs, to_funcs, V, i):
    for (f, t) in zip(from_funcs, to_funcs):
        t.vector()[:] = f[i, fem.dof_to_vertex_map(V)].astype('double')


def get_1d(func, V):
    return func.vector().get_local()[fem.vertex_to_dof_map(V)]


def save_plot(local_module_path, name):
    file = os.path.join(os.path.dirname(local_module_path), name)
    directory = os.path.dirname(os.path.abspath(file))
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(name)


def piecewise(mesh, subdomain, V0, *values):
    V0 = fem.FunctionSpace(mesh, 'DG', 0)
    k = fem.Function(V0)
    for cell in range(len(subdomain.array())):
        marker = subdomain.array()[cell]
        k.vector()[cell] = values[marker]

    # return k
    return fem.interpolate(k, fem.FunctionSpace(mesh, 'Lagrange', 1))


def piecewise2(V, *values):
    x = sym.Symbol('x[0]')
    E = sym.Piecewise((values[0], x <= 1.0 + fem.DOLFIN_EPS), (values[1], sym.And(x > 1.0, x < 2.0)),
                      (values[2], x >= 2.0 - fem.DOLFIN_EPS), (0, True))
    exp = fem.Expression(sym.printing.ccode(E), degree=0)
    fun = fem.interpolate(exp, V)

    return fun


def mkparam(markers, k_1=0, k_2=0, k_3=0, k_4=0):
    x = sym.Symbol('x[0]')
    asdf = sym.Piecewise((k_1, x <= 1.0 + fem.DOLFIN_EPS), (k_2, sym.And(x > 1.0, x < 2.0)),
                         (k_3, x >= 2.0 - fem.DOLFIN_EPS), (0, True))
    cppcode = """
    class K : public Expression
    {
        public:
            void eval(Array<double>& values,
                      const Array<double>& x,
                      const ufc::cell& cell) const
            {
                switch((*markers)[cell.index]){
                case 1:
                    values[0] = k_1;
                    break;
                case 2:
                    values[0] = k_2;
                    break;
                case 3:
                    values[0] = k_3;
                    break;
                case 4:
                    values[0] = k_4;
                    break;
                default:
                    values[0] = 0;
                }
            }

        std::shared_ptr<MeshFunction<std::size_t>> markers;
        double k_1, k_2, k_3, k_4;
    };
    """

    var = fem.Expression(cppcode=cppcode, degree=1)
    var.markers = markers
    var.k_1, var.k_2, var.k_3, var.k_4 = k_1, k_2, k_3, k_4
    # var = fem.Expression(sym.ccode(asdf), degree=1)
    return var


def overlay_plt(xdata, time, title, *ydata, figsize=(15, 9), linestyles=('-', '--')):
    fig, ax = plt.subplots(figsize=figsize)

    new_x = np.repeat([xdata], len(time), axis=0).T

    for i, data in enumerate(ydata):
        if i is 1:
            plt.plot(new_x, data.T, linestyles[i], marker='o')
        else:
            plt.plot(new_x, data.T, linestyles[i])
        plt.gca().set_prop_cycle(None)
    plt.grid(), plt.title(title)

    legend1 = plt.legend(['t = {}'.format(t) for t in time], title='Time', bbox_to_anchor=(1.01, 1), loc=2,
                         borderaxespad=0.)
    ax.add_artist(legend1)

    h = [plt.plot([], [], color="gray", ls=linestyles[i])[0] for i in range(len(linestyles))]
    plt.legend(handles=h, labels=["FEniCS", "COMSOL"], title="Solver", bbox_to_anchor=(1.01, 0), loc=3,
               borderaxespad=0.)


def norm_rmse(estimated, true):
    return engine.rmse(estimated, true) / (np.max(true) - np.min(true))


def report(mesh, time, estimated, true, name):
    rmse = norm_rmse(estimated, true)
    print('{name} normalized RMSE%:'.format(name=name))
    for i, t in enumerate(time):
        print('\tt = {time:3.1f}: {rmse:.3%}'.format(time=t, rmse=rmse[i]))

    overlay_plt(mesh, time, name, estimated, true)


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
