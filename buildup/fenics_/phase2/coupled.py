import sys
from functools import partial

import dolfin as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from buildup import common, utilities
from mtnlion.newman import equations


def picard_solver(a, lin, estimated, previous, bc, phis, phie):
    eps = 1.0
    eps_1 = 1.0
    tol = 1e-5
    iter = 0
    maxiter = 25
    while eps > tol and iter < maxiter:
        fem.solve(a == lin, estimated, bc)

        # calculate norm
        diff = estimated.vector().get_local() - previous.vector().get_local()
        eps = np.linalg.norm(diff, ord=np.Inf)

        print("iter={}, norm={}".format(iter, eps))

        if iter > 1 and eps > eps_1:
            print("Solution diverging, reverting and stopping.")
            estimated.assign(previous)
            break

        # update j
        phis.assign(estimated.split(True)[0])
        # phie.assign(estimated.split(True)[1])

        # set previous solution
        previous.assign(estimated)
        eps_1 = eps
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
    Acell = cmn.fenics_consts.Acell

    x = sym.Symbol("ce")
    y = sym.Symbol("x")
    kp = cmn.fenics_consts.kappa_ref.subs(y, x)

    dfdc = sym.Symbol("dfdc")
    # dfdc = 0
    kd = (
        fem.Constant(2)
        * cmn.fenics_consts.R
        * cmn.fenics_consts.Tref
        / cmn.fenics_consts.F
        * (fem.Constant(1) + dfdc)
        * (cmn.fenics_consts.t_plus - fem.Constant(1))
    )
    kappa_D = fem.Expression(sym.printing.ccode(kd), dfdc=0, degree=1)

    P = fem.FiniteElement("CG", cmn.mesh.ufl_cell(), degree=1)
    element = fem.MixedElement([P, P, P, P])
    V = fem.FunctionSpace(cmn.mesh, element)
    V_ = [V.sub(i).collapse() for i in range(4)]

    v = fem.TestFunction(V)
    v_phis, v_phie, v_ce, v_cse = fem.split(v)
    u = fem.TrialFunction(V)
    u_phis, u_phie, u_ce, u_cse = fem.split(u)
    bc = [fem.DirichletBC(V.sub(0), 0.0, domain.boundary_markers, 1), 0, 0]

    # true solutions
    true = fem.Function(V)

    # fenics solutions
    estimated = fem.Function(V)

    phis_f = fem.Function(domain.V)
    phie_f = fem.Function(domain.V)
    ce_f = fem.Function(domain.V)  # "previous solution"
    cse_f = fem.Function(domain.V)

    Uocp = equations.Uocp(cse_f, **cmn.fenics_params)
    j = equations.j(
        ce_f, cse_f, phie_f, phis_f, Uocp, **cmn.fenics_params, **cmn.fenics_consts, dm=domain.domain_markers
    )
    phis_form = partial(
        equations.phis, j, u_phis, v_phis, domain.dx((0, 2)), **cmn.fenics_params, **cmn.fenics_consts, ds=domain.ds(4)
    )
    phie_form = partial(equations.phie, j, ce_f, u_phie, v_phie, domain.dx, **cmn.fenics_params, **cmn.fenics_consts)

    # initialize matrix to save solution results
    phis_array = np.empty((len(time_in), len(comsol.mesh)))
    phie_array = np.empty((len(time_in), len(comsol.mesh)))
    j_array = np.empty((len(time_in), len(comsol.mesh)))

    dx = domain.dx
    assigner = fem.FunctionAssigner(V, [domain.V] * 4)
    assigner.assign(estimated, [phis_f, phie_f, ce_f, cse_f])

    k = 0
    for i in range(len(time_in)):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step
        cse_f.vector()[:] = comsol.data.cse[i][fem.dof_to_vertex_map(V_[0])].astype("double")
        ce_f.vector()[:] = comsol.data.ce[i][fem.dof_to_vertex_map(V_[1])].astype("double")
        phie_f.vector()[:] = comsol.data.phie[i][fem.dof_to_vertex_map(V_[2])].astype("double")
        phis_f.vector()[:] = comsol.data.phis[i_1][fem.dof_to_vertex_map(domain.V)].astype("double")

        bc[1] = fem.DirichletBC(V.sub(0), comsol.data.phis[i][-1], domain.boundary_markers, 4)
        bc[2] = fem.DirichletBC(V.sub(1), comsol.data.phie[i, 0], domain.boundary_markers, 1)

        kappa_ref = fem.Expression(sym.printing.ccode(kp), ce=ce_f, degree=1)
        kappa_eff = kappa_ref * cmn.fenics_params.eps_e ** cmn.fenics_params.brug_kappa
        kappa_Deff = kappa_D * kappa_ref * cmn.fenics_params.eps_e

        Feq = (
            phis_form(neumann=fem.Constant(Iapp[i]) / Acell)
            + phie_form(kappa_eff=kappa_eff, kappa_Deff=kappa_Deff)
            + fem.inner(u_ce, v_ce) * dx
            + fem.inner(u_cse, v_cse) * dx
            + fem.inner(u_phis, v_phis) * dx(1)
        )

        a = fem.lhs(Feq)
        lin = fem.rhs(Feq)

        picard_solver(a, lin, estimated, true, bc, phis_f, phie_f)

        phis_array[k, :] = estimated.split(True)[0].vector().get_local()[fem.vertex_to_dof_map(V_[0])]
        phie_array[k, :] = estimated.split(True)[1].vector().get_local()[fem.vertex_to_dof_map(V_[0])]
        j_array[k, :] = fem.interpolate(j, V_[0]).vector().get_local()[fem.vertex_to_dof_map(V_[0])]
        k += 1

    # x, ce, cse, phie, phis, csmax, ce0, alpha, k_norm_ref, F, R, Tref, Uocp_neg, Uocp_pos
    d = dict()
    d["x"] = comsol.mesh
    d["ce"] = comsol.data.ce[1::2]
    d["cse"] = comsol.data.cse[1::2]
    d["phie"] = comsol.data.phie[1::2]
    d["phis"] = phis_array

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
    dta = equations.eval_j(**neg, **cmn.consts)

    utilities.report(
        comsol.neg, time_in, phis_array[:, comsol.neg_ind], comsol.data.phis[:, comsol.neg_ind][1::2], "$\Phi_s^{neg}$"
    )
    plt.show()
    utilities.report(
        comsol.pos, time_in, phis_array[:, comsol.pos_ind], comsol.data.phis[:, comsol.pos_ind][1::2], "$\Phi_s^{pos}$"
    )
    plt.show()

    utilities.report(comsol.mesh, time_in, phie_array, comsol.data.phie[1::2], "$\Phi_e$")
    plt.show()
    utilities.report(comsol.neg, time_in, dta, comsol.data.j[:, comsol.neg_ind][1::2], "$j^{neg}$")
    plt.show()
    utilities.report(
        comsol.pos, time_in, j_array[:, comsol.pos_ind], comsol.data.j[:, comsol.pos_ind][1::2], "$j^{pos}$"
    )
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
