import fenics as fem
import matplotlib.pyplot as plt
import numpy as np

import common
import utilities


def ce():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    # time_in = [0.1, 5, 10, 15]
    time_in = np.arange(9, 11, 0.1)
    dt = 0.1
    dtc = fem.Constant(dt)
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    # Collect common data
    cmn = common.Common(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time_in), len(cmn.comsol_solution.mesh)))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm
    eps_e, de_eff, Lc, a_s = common.collect(cmn.fenics_params, 'eps_e', 'De_eff', 'L', 'a_s')
    t_plus = cmn.fenics_consts.t_plus

    # create function space and basis functions
    V = fem.FunctionSpace(mesh, 'Lagrange', 1)
    ce = fem.TrialFunction(V)
    ce_1 = fem.Function(V)
    jbar = fem.Function(V)
    v = fem.TestFunction(V)

    # Initialize Dirichlet BCs
    # bc = [fem.DirichletBC(V, comsol_sol.data.ce[i, 0], bm, 1), fem.DirichletBC(V, comsol_sol.data.ce[i, -1], bm, 4)]

    # Handle internal Neumann BCs
    boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=1)
    b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=2)
    b1.mark(boundary_markers, 2)
    b2.mark(boundary_markers, 3)
    n = fem.FacetNormal(mesh)
    dS = fem.Measure('dS', domain=mesh, subdomain_data=boundary_markers)

    # Variational problem
    neumann = dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce('-')), n('-')) * v('-') * dS(2) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce('+')), n('+')) * v('+') * dS(2) + \
              dtc * de_eff('-') / Lc('-') * fem.inner(fem.grad(ce('-')), n('-')) * v('-') * dS(3) + \
              dtc * de_eff('+') / Lc('+') * fem.inner(fem.grad(ce('+')), n('+')) * v('+') * dS(3)

    # a = Lc*eps_e*ce*v*dx
    # L = Lc*eps_e*ce_1*v*dx - dtc*de_eff/Lc*fem.dot(fem.grad(ce_1), fem.grad(v))*dx + dtc*Lc*a_s*\
    #     (fem.fenics_constsant(1) - t_plus)*jbar*v*dx + neumann

    a = Lc * eps_e * ce * v * dx + dtc * de_eff / Lc * fem.dot(fem.grad(ce), fem.grad(v)) * dx - neumann
    L = -dtc * Lc * a_s * (fem.Constant(1) - t_plus) * jbar * v * dx + Lc * eps_e * ce_1 * v * dx

    ce = fem.Function(V)
    k = 0
    for i, t in enumerate(time_in):
        i_1 = i * 2  # previous time step
        i = i * 2 + 1  # current time step

        jbar.vector()[:] = comsol_sol.data.j[i, fem.dof_to_vertex_map(V)].astype('double')
        if k is 0:
            ce_1.vector()[:] = comsol_sol.data.ce[i_1, fem.dof_to_vertex_map(V)].astype('double')
        else:
            ce_1.assign(ce)
            # ce_1.vector()[:] = comsol_sol.data.ce[i_1, fem.dof_to_vertex_map(V)].astype('double')

        bc = [fem.DirichletBC(V, comsol_sol.data.ce[i, 0], bm, 1), fem.DirichletBC(V, comsol_sol.data.ce[i, -1], bm, 4)]
        # Solve
        fem.solve(a == L, ce, bc)

        u_nodal_values = ce.vector()
        u_array[k, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]
        # print('t={time}: error = {error}'.format(time=time[i], error=np.abs(u_array[k, :] - comsol_sol.data.ce[i, :]).max()))
        k += 1

    # utilities.report(comsol_sol.mesh, time_in, u_array, comsol_sol.data.ce[1::2], '$run$')
    utilities.overlay_plt(comsol_sol.mesh, time_in, 'ce', u_array)

    plt.savefig('comsol_compare_ce.png')
    plt.show()


if __name__ == '__main__':
    ce()
