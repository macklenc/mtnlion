import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import utilities


def phi_e():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(cmn.comsol_solution.mesh)))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm
    # sigma_eff, Lc, a_s, F, eps_e, t_plus = cmn.sigma_eff, cmn.Lc, cmn.a_s, cmn.F, cmn.eps_e, cmn.t_plus
    Lc, a_s, eps_e, sigma_eff, brug_kappa = common.collect(cmn.params, 'L', 'a_s', 'eps_e', 'sigma_eff', 'brug_kappa')
    F, t_plus, R, T = common.collect(cmn.const, 'F', 't_plus', 'R', 'Tref')

    x = sym.Symbol('ce')
    y = sym.Symbol('x')
    kp = cmn.const.kappa_ref[0].subs(y, x)

    dfdc = sym.Symbol('dfdc')
    # dfdc = 0
    kd = fem.Constant(2) * R * T / F * (fem.Constant(1) + dfdc) * (t_plus - fem.Constant(1))
    kappa_D = fem.Expression(sym.printing.ccode(kd), dfdc=0, degree=1)
    # func = sym.lambdify(x, kp, 'numpy')
    # plt.plot(np.arange(0, 3000.1, 0.1), func(np.arange(0, 3000.1, 0.1)))
    # print(max( func(np.arange(0, 3000.1, 0.1))))
    # plt.grid()
    # plt.show()

    for i, j in enumerate(comsol_sol.data.j):
        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)
        phie = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        # Initialize Dirichlet BCs
        bc = [fem.DirichletBC(V, comsol_sol.data.phie[i, 0], bm, 1)]
        jbar = fem.Function(V)
        jbar.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double')
        ce = fem.Function(V)
        ce.vector()[:] = comsol_sol.data.ce[i, fem.dof_to_vertex_map(V)].astype('double')

        # calculate kappa
        kappa_ref = fem.Expression(sym.printing.ccode(kp), ce=ce, degree=1)
        kappa_eff = kappa_ref * eps_e ** brug_kappa
        kappa_Deff = kappa_D*kappa_ref*eps_e

        boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=1)
        b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=2)
        b1.mark(boundary_markers, 2)
        b2.mark(boundary_markers, 3)

        # Setup measures
        # dS = fem.Measure('dS', domain=mesh, subdomain_data=boundary_markers)

        # Setup Neumann BCs
        # neumann = -cmn.Iapp[i]/cmn.Acell*fem.avg(v)*dS(2)-cmn.Iapp[i]/cmn.Acell*fem.avg(v)*dS(3)
        # neumann = ((-cmn.Iapp[i] / cmn.Acell * v('-') - cmn.Iapp[i] / cmn.Acell * v('+')) * dS(2)
        #            + (-cmn.Iapp[i] / cmn.Acell * v('-') - cmn.Iapp[i] / cmn.Acell * v('+')) * dS(3))

        # neumann = -cmn.Iapp[i]/cmn.Acell*v/Lc*ds(2)-cmn.Iapp[i]/cmn.Acell*v/Lc*ds(3)
        # (-cmn.Iapp[i]/cmn.Acell*v('-')+cmn.Iapp[i]/cmn.Acell*v('+'))*dS(2)\
        #       +cmn.Iapp[i]/cmn.Acell*v('-')*dS(3)-cmn.Iapp[i]/cmn.Acell*v('+')*dS(3)

        # mod = utilities.mkparam(dm, 0.6, 0.85, 0.67)
        # Setup equation
        a = kappa_eff/Lc*fem.dot(fem.grad(phie), fem.grad(v))*dx

        L = Lc * a_s * F * jbar * v * dx - kappa_Deff / Lc * \
            fem.dot(fem.grad(fem.ln(ce)), fem.grad(v)) * dx  # + neumann

        # Solve
        phie = fem.Function(V)
        fem.solve(a == L, phie, bc)
        u_nodal_values = phie.vector()
        u_array[i, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]

    # utilities.overlay_plt(comsol_sol.mesh, time, '$\Phi_e$', u_array, comsol_sol.data.phie)
    # print(engine.rmse(u_array, comsol_sol.data.phie))
    utilities.report(comsol_sol.mesh, time, u_array, comsol_sol.data.phie, '$\Phi_e$')

    plt.savefig('comsol_compare_phie.png')
    plt.show()

if __name__ == '__main__':
    phi_e()
