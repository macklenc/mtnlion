import os
import time

import dolfin as fem
import matplotlib.pyplot as plt
import munch
import numpy as np
import sympy as sym
from scipy import interpolate

import mtnlion.comsol as comsol
import mtnlion.engine as engine
import mtnlion.loader as loader


def gather_data():
    # Load required cell data
    localdir = os.path.dirname(__file__)
    resources = os.path.join(localdir, "reference/")
    params = engine.fetch_params(os.path.join(resources, "GuAndWang_parameter_list.xlsx"))
    d_comsol = comsol.load(os.path.join(resources, "guwang_hifi.npz"))
    pseudo_mesh_file = os.path.join(resources, "comsol_solution/hifi/cs.xml")
    Uocp_spline = loader.load_numpy_file(os.path.join(resources, "Uocp_spline.npz"))
    input_current = loader.load_csv_file(os.path.join(resources, "comsol_solution/hifi/input_current.csv.bz2"))
    return d_comsol, params, pseudo_mesh_file, Uocp_spline, input_current


def gather_expressions():
    # TODO: read entire directory
    localdir = os.path.dirname(__file__)
    code = dict()
    with open(os.path.join(localdir, "../mtnlion/headers/xbar.h")) as f:
        code["xbar"] = "".join(f.readlines())

    with open(os.path.join(localdir, "../mtnlion/headers/composition.h")) as f:
        code["composition"] = "".join(f.readlines())

    with open(os.path.join(localdir, "../mtnlion/headers/piecewise.h")) as f:
        code["piecewise"] = "".join(f.readlines())

    with open(os.path.join(localdir, "../mtnlion/headers/xbar_simple.h")) as f:
        code["xbar_simple"] = "".join(f.readlines())

    return munch.Munch(code)


# TODO: this is ugly
expressions = gather_expressions()


def create_solution_matrices(nr, nc, r):
    return tuple(np.empty((nr, nc)) for _ in range(r))


def create_functions(V, r):
    return tuple(fem.Function(V) for _ in range(r))


def assign_functions(from_funcs, to_funcs, V, i):
    for (f, t) in zip(from_funcs, to_funcs):
        t.vector()[:] = f[i, fem.dof_to_vertex_map(V)].astype("double")


def get_1d(func, V):
    return func.vector().get_local()[fem.vertex_to_dof_map(V)]


def save_plot(local_module_path, name):
    file = os.path.join(os.path.dirname(local_module_path), name)
    directory = os.path.dirname(os.path.abspath(file))
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(name)


def piecewise(mesh, subdomain, V0, *values):
    # TODO: benchmark differences
    # same thing as commented code below
    # h = fem.Expression('x[0] <= 1.0 ? f : (x[0] >= 2 ? g : h)', f=values[0], h=values[1], g=values[2], degree=1)
    # k = fem.interpolate(h, V0)

    V0 = fem.FunctionSpace(mesh, "DG", 0)
    k = fem.Function(V0)
    for cell in range(len(subdomain.array())):
        marker = subdomain.array()[cell]
        k.vector()[cell] = values[marker]

    return k


def fenics_interpolate(xy_values, cell_type="Lagrange", degree=1):
    x_values = xy_values[:, 0]
    y_values = xy_values[:, 1]

    mesh = fem.IntervalMesh(len(x_values) - 1, 0, 3)  # length doesn't matter
    mesh.coordinates()[:] = np.array([x_values]).transpose()

    V1 = fem.FunctionSpace(mesh, cell_type, degree)
    interp = fem.Function(V1)
    interp.vector()[:] = y_values[fem.vertex_to_dof_map(V1)]

    return interp


def interp_time(time, data):
    y = interpolate.interp1d(time, data, axis=0, fill_value="extrapolate")
    return y


def find_cse_from_cs(comsol):
    data = np.append(comsol.pseudo_mesh, comsol.data.cs.T, axis=1)  # grab cs for each time
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]  # find the indices of the solution where r=1 (cse)
    data = data[indices]  # reduce data set to only show cse
    data = data[data[:, 0].argsort()]  # organize the coordinates for monotonicity
    xcoor = data[:, 0]  # x coordinates are in the first column, y should always be 1 now
    neg_ind = np.where(xcoor <= 1)[0]  # using the pseudo dims definition of neg and pos electrodes
    pos_ind = np.where(xcoor >= 1.5)[0]
    cse = data[:, 2:]  # first two columns are the coordinates

    return xcoor, cse.T, neg_ind, pos_ind


# TODO: add builder method for creating expression wrappers
def compose(inner, outer, degree=1):
    return fem.CompiledExpression(
        fem.compile_cpp_code(expressions.composition).Composition(),
        inner=inner.cpp_object(),
        outer=outer.cpp_object(),
        degree=degree,
    )


def piecewise2(V, *values):
    x = sym.Symbol("x[0]")
    E = sym.Piecewise(
        (values[0], x <= 1.0 + fem.DOLFIN_EPS),
        (values[1], sym.And(x > 1.0, x < 2.0)),
        (values[2], x >= 2.0 - fem.DOLFIN_EPS),
        (0, True),
    )
    exp = fem.Expression(sym.printing.ccode(E), degree=0)
    fun = fem.interpolate(exp, V)

    return fun


def mkparam(markers, k_1=fem.Constant(0), k_2=fem.Constant(0), k_3=fem.Constant(0), k_4=fem.Constant(0)):
    var = fem.CompiledExpression(fem.compile_cpp_code(expressions.piecewise).Piecewise(), degree=1)
    var.markers = markers
    # NOTE: .cpp_object() will not be required later as per
    # https://bitbucket.org/fenics-project/dolfin/issues/1041/compiledexpression-cant-be-initialized
    var.k_1, var.k_2, var.k_3, var.k_4 = k_1.cpp_object(), k_2.cpp_object(), k_3.cpp_object(), k_4.cpp_object()
    return var


def overlay_plt(xdata, time, title, *ydata, figsize=(15, 9), linestyles=("-", "--")):
    fig, ax = plt.subplots(figsize=figsize)

    new_x = np.repeat([xdata], len(time), axis=0).T

    for i, data in enumerate(ydata):
        if i is 1:
            plt.plot(new_x, data.T, linestyles[i], marker="o")
        else:
            plt.plot(new_x, data.T, linestyles[i])
        plt.gca().set_prop_cycle(None)
    plt.grid(), plt.title(title)

    legend1 = plt.legend(
        ["t = {}".format(t) for t in time], title="Time", bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0
    )
    ax.add_artist(legend1)

    h = [plt.plot([], [], color="gray", ls=linestyles[i])[0] for i in range(len(linestyles))]
    plt.legend(
        handles=h, labels=["FEniCS", "COMSOL"], title="Solver", bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0.0
    )


def norm_rmse(estimated, true):
    return engine.rmse(estimated, true) / (np.max(true) - np.min(true))


def report(mesh, time, estimated, true, name):
    rmse = norm_rmse(estimated, true)
    print("{name} normalized RMSE%:".format(name=name))
    for i, t in enumerate(time):
        print("\tt = {time:3.1f}: {rmse:.3%}".format(time=t, rmse=rmse[i]))

    overlay_plt(mesh, time, name, estimated, true)


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def picard_solver(F, u, u_, bc, tol=1e-5, maxiter=25):
    eps = 1.0
    iter = 0
    while eps > tol and iter < maxiter:
        iter += 1
        fem.solve(F, u, bc)
        diff = u.vector().get_local() - u_.vector().get_local()
        eps = np.linalg.norm(diff, ord=np.Inf)
        print("iter={}, norm={}".format(iter, eps))
        u_.assign(u)


# Doesn't work!
def newton_solver(F, u_, bc, J, V, a_tol=1e-7, r_tol=1e-10, maxiter=25, relaxation=1):
    eps = 1.0
    iter = 0
    u_inc = fem.Function(V)
    while eps > a_tol and iter < maxiter:
        iter += 1
        # plt.plot(get_1d(u_, V))
        # plt.show()

        A, b = fem.assemble_system(J, -F, bc)
        fem.solve(A, u_inc.vector(), b)
        eps = np.linalg.norm(u_inc.vector().get_local(), ord=np.Inf)
        print("iter={}, eps={}".format(iter, eps))

        # plt.plot(get_1d(u_inc, V))
        # plt.show()

        a = fem.assemble(F)
        # for bnd in bc:
        bc.apply(a)
        print("b.norm = {}, linalg norm = {}".format(b.norm("l2"), np.linalg.norm(a.get_local(), ord=2)))
        fnorm = b.norm("l2")

        u_.vector()[:] += relaxation * u_inc.vector()

        print("fnorm: {}".format(fnorm))


def generate_test_stats(time, indices, estimated, true):
    estimated = estimated(time)
    true = true(time)
    rmse = np.array(
        [
            norm_rmse(estimated[:, indices.neg_ind], true[:, indices.neg_ind]),
            norm_rmse(estimated[:, indices.sep_ind], true[:, indices.sep_ind]),
            norm_rmse(estimated[:, indices.pos_ind], true[:, indices.pos_ind]),
        ]
    )

    mean = np.array(
        [
            np.mean(estimated[:, indices.neg_ind]),
            np.mean(estimated[:, indices.sep_ind]),
            np.mean(estimated[:, indices.pos_ind]),
        ]
    )

    std = np.array(
        [
            np.std(estimated[:, indices.neg_ind]),
            np.std(estimated[:, indices.sep_ind]),
            np.std(estimated[:, indices.pos_ind]),
        ]
    )
    return rmse, mean, std
