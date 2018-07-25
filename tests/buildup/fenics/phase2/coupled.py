import sys
from functools import partial

import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import utilities
from mtnlion.newman import equations


def picard_solver(a, lin, phis, phis_, bc):
    eps = 1.0
    tol = 1e-5
    iter = 0
    maxiter = 25
    while eps > tol and iter < maxiter:
        iter += 1
        fem.solve(a == lin, phis, bc)
        phis.vector()[np.isnan(phis.vector().get_local())] = 0
        diff = phis.split(True)[0].vector().get_local() - phis_.vector().get_local()
        eps = np.linalg.norm(diff, ord=np.Inf)
        print('iter={}, norm={}'.format(iter, eps))
        phis_.assign(phis.split(True)[0])


def main():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    I_1C = 20.5
    Iapp = [I_1C if 10 <= i <= 20 else -I_1C if 30 <= i <= 40 else 0 for i in time]

    # Collect common data
    cmn = common.Common(time)
    domain = cmn.domain
    comsol = cmn.comsol_solution
    Acell = cmn.const.Acell

    P = fem.FiniteElement('CG', cmn.mesh.ufl_cell(), degree=1)
    element = fem.MixedElement([P, P, P, P])
    V = fem.FunctionSpace(cmn.mesh, element)
    V_ = [V.sub(i).collapse() for i in range(4)]
    A_ = [fem.FunctionAssigner(V.sub(i), V_[i]) for i in range(4)]
    V1 = V.sub(0).collapse()
    V2 = V.sub(1).collapse()
    V3 = V.sub(2).collapse()
    V4 = V.sub(3).collapse()
    A1 = fem.FunctionAssigner(V.sub(0), V1)
    A2 = fem.FunctionAssigner(V.sub(1), V2)
    A3 = fem.FunctionAssigner(V.sub(2), V3)
    A4 = fem.FunctionAssigner(V.sub(3), V4)

    # v_phis, v_phie, v_ce, v_cse = fem.TestFunctions(V)
    v = fem.TestFunction(V)
    v_phis, v_phie, v_ce, v_cse = fem.split(v)
    # (u_phis, u_phie, u_ce, u_cse) = fem.TrialFunctions(V)
    u = fem.TrialFunction(V)
    u_phis, u_phie, u_ce, u_cse = fem.split(u)
    bc = [fem.DirichletBC(V.sub(0), 0.0, domain.boundary_markers, 1), 0]

    # true solutions
    true = fem.Function(V)
    # phis_f, phie_f, ce_f, cse_f = fem.split(true)
    # cse_f, ce_f, phis_f, phie_f, phis = (fem.Function(W) for _ in range(5))

    # fenics solutions
    estimated = fem.Function(V)
    # phis, phie, ce, cse = fem.split(estimated)
    phis = fem.Function(V_[0])

    true_funcs = [fem.Function(V_[i]) for i in range(4)]
    # for i in range(4):
    #     A_[i].assign(true.sub(i), true_funcs[i])
    # phis_f, phie_f, ce_f, cse_f = copy.copy(true_funcs)
    phis_f = fem.Function(V1)
    phie_f = fem.Function(V2)
    ce_f = fem.Function(V3)  # "previous solution"
    cse_f = fem.Function(V4)
    j = equations.j(ce_f, cse_f, phie_f, phis_f, **cmn.params, **cmn.const)
    phis_form = partial(equations.phis, j, u_phis, v_phis, domain.dx((0, 2)),
                        **cmn.params, **cmn.const, ds=domain.ds(4), nonlin=True)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(comsol.mesh)))
    u_array2 = np.empty((len(time), len(comsol.mesh)))

    A1.assign(true.sub(0), phis_f)
    A2.assign(true.sub(1), phie_f)
    A3.assign(true.sub(2), ce_f)
    A4.assign(true.sub(3), cse_f)

    comsol_data = [comsol.data.phis, comsol.data.phie, comsol.data.ce, comsol.data.cse]
    dx = domain.dx
    comsol_data = zip(comsol.data.cse, comsol.data.ce, comsol.data.phis, comsol.data.phie)
    for i, (cse_t, ce_t, phis_t, phie_t) in enumerate(comsol_data):
        cse_f.vector()[:] = cse_t[fem.dof_to_vertex_map(V_[0])].astype('double')
        ce_f.vector()[:] = ce_t[fem.dof_to_vertex_map(V_[1])].astype('double')
        phie_f.vector()[:] = phie_t[fem.dof_to_vertex_map(V_[2])].astype('double')
        phis_f.vector()[:] = phis_t[fem.dof_to_vertex_map(V_[3])].astype('double')
        # for space in range(4):
        #     func = fem.Function(V_[space])
        #     func.vector()[:] = solution[space][fem.dof_to_vertex_map(V_[space])].astype('double')
        #     A_[space].assign(true.sub(space), func)
        # for space in range(4):
        #     true_funcs[i].vector()[:] = solution[space][fem.dof_to_vertex_map(V_[space])].astype('double')
        # A_[i].assign(true.sub(i), true_funcs[i])

        jeq = fem.interpolate(j, V_[0]).vector().get_local()

        bc[1] = fem.DirichletBC(V.sub(0), phis_t[-1], domain.boundary_markers, 4)
        Feq = phis_form(neumann=fem.Constant(Iapp[i]) / Acell) + fem.inner(u_phie, v_phie) * dx + fem.inner(u_ce,
                                                                                                            v_ce) * dx + fem.inner(
            u_cse, v_cse) * dx

        a = fem.lhs(Feq)
        lin = fem.rhs(Feq)

        picard_solver(a, lin, estimated, phis_f, bc)

        u_array[i, :] = estimated.split(True)[0].vector().get_local()[fem.vertex_to_dof_map(V1)]
        u_array2[i, :] = fem.interpolate(j, V1).vector().get_local()[fem.vertex_to_dof_map(V1)]

    utilities.report(comsol.neg, time, u_array[:, comsol.neg_ind],
                     comsol.data.phis[:, comsol.neg_ind], '$\Phi_s^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time, u_array[:, comsol.pos_ind],
                     comsol.data.phis[:, comsol.pos_ind], '$\Phi_s^{pos}$')
    plt.show()

    utilities.report(comsol.neg, time, u_array2[:, comsol.neg_ind],
                     comsol.data.j[:, comsol.neg_ind], '$j^{neg}$')
    plt.show()
    utilities.report(comsol.pos, time, u_array2[:, comsol.pos_ind],
                     comsol.data.j[:, comsol.pos_ind], '$j^{pos}$')
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
