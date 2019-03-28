import sys

import dolfin as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from buildup import common, utilities
from mtnlion.newman import equations

# NOTE: Deprecated

def picard_solver(a, lin, estimated, previous, bc):
    eps = 1.0
    tol = 1e-5
    iter = 0
    maxiter = 25
    while eps > tol and iter < maxiter:
        fem.solve(a == lin, estimated, bc)

        # calculate norm
        diff = estimated.vector().get_local() - previous.vector().get_local()
        eps = np.linalg.norm(diff, ord=np.Inf)

        print("iter={}, norm={}".format(iter, eps))

        # set previous solution
        previous.assign(estimated)
        iter += 1


def main():
    # Times at which to run solver
    time_in = [0.1, 5, 9.9, 10, 10.1, 15, 20]

    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]

    # Collect common data
    cmn = common.Common(time)
    domain = cmn.domain
    comsol = cmn.comsol_solution
    k_norm_ref, csmax, alpha, L, a_s, sigma_eff = common.collect(
        cmn.fenics_params, "k_norm_ref", "csmax", "alpha", "L", "a_s", "sigma_eff"
    )
    F, R, Tref, ce0, Acell = common.collect(cmn.fenics_consts, "F", "R", "Tref", "ce0", "Acell")
    Lc, a_s, eps_e, sigma_eff, brug_kappa = common.collect(
        cmn.fenics_params, "L", "a_s", "eps_e", "sigma_eff", "brug_kappa"
    )
    F, t_plus, R, T = common.collect(cmn.fenics_consts, "F", "t_plus", "R", "Tref")

    x = sym.Symbol("ce")
    y = sym.Symbol("x")
    kp = cmn.fenics_consts.kappa_ref.subs(y, x)

    dfdc = sym.Symbol("dfdc")
    # dfdc = 0
    kd = fem.Constant(2) * R * T / F * (fem.Constant(1) + dfdc) * (t_plus - fem.Constant(1))
    kappa_D = fem.Expression(sym.printing.ccode(kd), dfdc=0, degree=1)

    V = domain.V
    v = fem.TestFunction(V)
    du = fem.TrialFunction(V)

    cse_f = fem.Function(V)
    ce_f = fem.Function(V)
    phis_f = fem.Function(V)
    jbar = fem.Function(V)
    phie_f = fem.Function(V)  # "previous solution"
    phie = fem.Function(V)  # "previous solution"

    kappa_ref = fem.Expression(sym.printing.ccode(kp), ce=ce_f, degree=1)
    kappa_eff = kappa_ref * eps_e ** brug_kappa
    kappa_Deff = kappa_D * kappa_ref * eps_e

    Uocp = equations.Uocp(cse_f, **cmn.fenics_params)
    j = equations.j(
        ce_f, cse_f, phie_f, phis_f, Uocp, csmax, ce0, alpha, k_norm_ref, F, R, Tref, dm=domain.domain_markers
    )
    # phie(jbar, ce, phie, v, dx, L, a_s, F, kappa_eff, kappa_Deff, ds=0, neumann=0, nonlin=False, **kwargs):

    u = fem.TrialFunction(V)
    v = fem.TestFunction(V)
    lhs, rhs = equations.phie(j, ce_f, u, v, kappa_eff, kappa_Deff, **cmn.fenics_params, **cmn.fenics_consts)
    phie_form = (lhs - rhs) * domain.dx

    # initialize matrix to save solution results
    u_array = np.empty((len(time_in), len(comsol.mesh)))
    u_array2 = np.empty((len(time_in), len(comsol.mesh)))

    k = 0
    for i in range(len(time_in)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step

        cse_f.vector()[:] = comsol.data.cse[i][fem.dof_to_vertex_map(V)].astype("double")
        ce_f.vector()[:] = comsol.data.ce[i][fem.dof_to_vertex_map(V)].astype("double")
        phie_f.vector()[:] = comsol.data.phie[i][fem.dof_to_vertex_map(V)].astype("double")
        phis_f.vector()[:] = comsol.data.phis[i][fem.dof_to_vertex_map(V)].astype("double")
        jbar.vector()[:] = comsol.data.j[i][fem.dof_to_vertex_map(V)].astype("double")

        bc = [fem.DirichletBC(V, comsol.data.phie[i, 0], domain.boundary_markers, 1)]
        Feq = phie_form

        # fem.solve(fem.lhs(Feq) == fem.rhs(Feq), phie_f, bc)
        phie.assign(phie_f)
        picard_solver(fem.lhs(Feq), fem.rhs(Feq), phie_f, phie, bc)

        # J = fem.derivative(Feq, phie_f, du)
        # problem = fem.NonlinearVariationalProblem(Feq, phis_f, bc, J)
        # solver = fem.NonlinearVariationalSolver(problem)
        #
        # prm = solver.parameters
        # prm['newton_solver']['absolute_tolerance'] = 1e-8
        # prm['newton_solver']['relative_tolerance'] = 1e-7
        # prm['newton_solver']['maximum_iterations'] = 25
        # prm['newton_solver']['relaxation_parameter'] = 1.0
        # solver.solve()

        u_array[k, :] = phie_f.vector().get_local()[fem.vertex_to_dof_map(domain.V)]
        u_array2[k, :] = fem.interpolate(j, V).vector().get_local()[fem.vertex_to_dof_map(V)]
        k += 1

    d = dict()
    d["x"] = comsol.mesh
    d["ce"] = comsol.data.ce[1::2]
    d["cse"] = comsol.data.cse[1::2]
    d["phie"] = u_array
    d["phis"] = comsol.data.phis[1::2]

    neg_params = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in cmn.params.items()}
    d = dict(d, **neg_params)

    def filter(x, sel="neg"):
        if sel is "neg":
            ind0 = 0
            ind1 = cmn.comsol_solution.neg_ind
        else:
            ind0 = 2
            ind1 = cmn.comsol_solution.pos_ind

        if isinstance(x, list):
            return x[ind0]

        if isinstance(x, np.ndarray):
            if len(x.shape) > 1:
                return x[:, ind1]

            return x[ind1]

        return x

    neg = dict(map(lambda x: (x[0], filter(x[1], "neg")), d.items()))
    # dta = equations.eval_j(**neg, **cmn.consts)

    utilities.report(comsol.mesh, time_in, u_array, comsol.data.phie[1::2], "$\Phi_e$")
    plt.show()

    # utilities.report(comsol.neg, time_in, dta,
    #                  comsol.data.j[:, comsol.neg_ind][1::2], '$j^{neg}$')
    # plt.show()
    utilities.report(
        comsol.pos, time_in, u_array2[:, comsol.pos_ind], comsol.data.j[:, comsol.pos_ind][1::2], "$j^{pos}$"
    )
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
