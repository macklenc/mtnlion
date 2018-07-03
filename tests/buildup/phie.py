import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import utilities


def phie():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect common data
    cmn = common.Common(time)

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(cmn.comsol_solution.mesh)))

    # create local variables
    comsol_sol = cmn.comsol_solution
    mesh, dx, ds, bm, dm = cmn.mesh, cmn.dx, cmn.ds, cmn.bm, cmn.dm
    sigma_eff, Lc, a_s, F, eps_e, t_plus = cmn.sigma_eff, cmn.Lc, cmn.a_s, cmn.F, cmn.eps_e, cmn.t_plus

    x = sym.Symbol('ce')
    kp = 100 * (4.1253e-4 + 5.007 * x * 1e-6 - 4.7212e3 * x ** 2 * 1e-12 +
                1.5094e6 * x ** 3 * 1e-18 - 1.6018e8 * x ** 4 * 1e-24)

    dfdc = sym.Symbol('dfdc')
    # dfdc = 0
    kd = fem.Constant(2) * cmn.R * cmn.T / cmn.F * (fem.Constant(1) + dfdc) * (t_plus - fem.Constant(1))
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
        kappa_eff = kappa_ref * eps_e
        kappa_Deff = kappa_D*kappa_ref*eps_e

        boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=1)
        b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=2)
        b1.mark(boundary_markers, 2)
        b2.mark(boundary_markers, 3)

        n = fem.FacetNormal(mesh)

        # Setup measures
        dS = fem.Measure('dS', domain=mesh, subdomain_data=boundary_markers)

        # Setup Neumann BCs
        # neumann = -cmn.Iapp[i]/cmn.Acell*fem.avg(v)*dS(2)-cmn.Iapp[i]/cmn.Acell*fem.avg(v)*dS(3)
        neumann = ((-cmn.Iapp[i] / cmn.Acell * v('-') - cmn.Iapp[i] / cmn.Acell * v('+')) * dS(2)
                   + (-cmn.Iapp[i] / cmn.Acell * v('-') - cmn.Iapp[i] / cmn.Acell * v('+')) * dS(3))

        # neumann = -cmn.Iapp[i]/cmn.Acell*v/Lc*ds(2)-cmn.Iapp[i]/cmn.Acell*v/Lc*ds(3)
        # (-cmn.Iapp[i]/cmn.Acell*v('-')+cmn.Iapp[i]/cmn.Acell*v('+'))*dS(2)\
        #       +cmn.Iapp[i]/cmn.Acell*v('-')*dS(3)-cmn.Iapp[i]/cmn.Acell*v('+')*dS(3)

        mod = utilities.mkparam(dm, 0.6, 0.85, 0.67)
        # Setup equation
        a = kappa_eff/Lc*fem.dot(fem.grad(phie), fem.grad(v))*dx

        L = Lc * a_s * F * jbar * v * dx - kappa_Deff / (Lc) * fem.dot(fem.grad(fem.ln(ce)),
                                                                       fem.grad(v)) * dx  # + neumann

        # Solve
        phie = fem.Function(V)
        fem.solve(a == L, phie, bc)
        u_nodal_values = phie.vector()
        u_array[i, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]

    utilities.overlay_plt(comsol_sol.mesh, time, '$\Phi_e$', u_array, comsol_sol.data.phie)

    plt.savefig('comsol_compare_phie.png')
    plt.show()

if __name__ == '__main__':
    phie()
