import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import domain2
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import utilities


def phie():
    # Times at which to run solver
    time = [0, 5, 10, 15, 20]

    # Collect required data
    comsol_data, params = utilities.gather_data()
    time_ind = engine.find_ind(comsol_data.time_mesh, time)
    data = comsol.get_standardized(comsol_data.filter_time(time_ind))

    # initialize matrix to save solution results
    u_array = np.empty((len(time), len(data.mesh)))
    mesh, dx, ds, bm, dm = domain2.generate_domain(data.mesh)

    # Initialize parameters
    F = fem.Constant(96485)
    R = fem.Constant(8.314) # universal gas constant
    T = fem.Constant(298.15)

    x = sym.Symbol('ce')
    kp = 100 * (4.1253e-4 + 5.007 * x * 1e-6 - 4.7212e3 * x ** 2 * 1e-12 +
                1.5094e6 * x ** 3 * 1e-18 - 1.6018e8 * x ** 4 * 1e-24)

    dfdc = sym.Symbol('dfdc')
    # dfdc = 0
    kd = fem.Constant(2) * R * T / F * (fem.Constant(1) + dfdc) * (fem.Constant(params.const.t_plus) - fem.Constant(1))
    kappa_D = fem.Expression(sym.printing.ccode(kd), dfdc=0, degree=1)
    # func = sym.lambdify(x, kp, 'numpy')
    # plt.plot(np.arange(0, 3000.1, 0.1), func(np.arange(0, 3000.1, 0.1)))
    # print(max( func(np.arange(0, 3000.1, 0.1))))
    # plt.grid()
    # plt.show()

    Lc = utilities.mkparam(dm, params.neg.L, params.sep.L, params.pos.L)
    eps_e = utilities.mkparam(dm, params.neg.eps_e ** params.neg.brug_kappaD,
                    params.sep.eps_e ** params.sep.brug_kappaD,
                    params.pos.eps_e ** params.pos.brug_kappaD)

    a_s = utilities.mkparam(dm, 3 * params.neg.eps_s / params.neg.Rs, 0, 3 * params.pos.eps_s / params.pos.Rs)

    for i, j in enumerate(data.data.j):
        # Define function space and basis functions
        V = fem.FunctionSpace(mesh, 'Lagrange', 1)
        phie = fem.TrialFunction(V)
        v = fem.TestFunction(V)

        # Initialize Dirichlet BCs
        bc = [fem.DirichletBC(V, data.data.phie[i, 0], bm, 1), fem.DirichletBC(V, data.data.phie[i, -1], bm, 4)]
        jbar = fem.Function(V)
        jbar.vector()[:] = j[fem.dof_to_vertex_map(V)].astype('double')
        ce = fem.Function(V)
        ce.vector()[:] = data.data.ce[i, fem.dof_to_vertex_map(V)].astype('double')

        # calculate kappa
        kappa_ref = fem.Expression(sym.printing.ccode(kp), ce=ce, degree=1)
        kappa_eff = kappa_ref * eps_e
        kappa_Deff = kappa_D*kappa_ref*eps_e

        # Setup equation
        a = kappa_eff/Lc*fem.dot(fem.grad(phie), fem.grad(v))*dx

        L = Lc*a_s*F*jbar*v*dx - kappa_Deff/(Lc)*fem.dot(fem.grad(fem.ln(ce)), fem.grad(v))*dx

        # Solve
        phie = fem.Function(V)
        fem.solve(a == L, phie, bc)
        u_nodal_values = phie.vector()
        u_array[i, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(V)]

    utilities.overlay_plt(data.mesh, time, '$\Phi_e$', u_array, data.data.phie)

    plt.savefig('comsol_compare_phie.png')
    plt.show()

if __name__ == '__main__':
    phie()
