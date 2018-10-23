import fenics as fem
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import common
import utilities

x_conv = '''
class XConv : public Expression
{
public:
  XConv() : Expression() {}

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
      switch((*markers)[c.index]){
        case 0:
            neg->eval(values, x, c);
            return;
        case 1:
            sep->eval(values, x, c);
            return;
        case 2:
            pos->eval(values, x, c);
            return;
      }
  }
  
  std::shared_ptr<MeshFunction<std::size_t>> markers;
  std::shared_ptr<GenericFunction> neg;
  std::shared_ptr<GenericFunction> sep;
  std::shared_ptr<GenericFunction> pos;
};
'''


composition = '''
class Composition : public Expression
{
public:
  Composition() : Expression() {}

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
      Array<double> val(3);
      inner->eval(val, x, c);
      outer->eval(values, val, c);      
  }
  
  std::shared_ptr<GenericFunction> outer;
  std::shared_ptr<GenericFunction> inner;
};
'''


def find_cse_from_cs(comsol):
    data = np.append(comsol.pseudo_mesh, comsol.data.cs[1::2].T, axis=1)  # grab cs for each time
    indices = np.where(np.abs(data[:, 1] - 1.0) <= 1e-5)[0]  # find the indices of the solution where r=1 (cse)
    data = data[indices]  # reduce data set to only show cse
    data = data[data[:, 0].argsort()]  # organize the coordinates for monotonicity
    xcoor = data[:, 0]  # x coordinates are in the first column, y should always be 1 now
    neg_ind = np.where(xcoor <= 1)[0]  # using the pseudo dims definition of neg and pos electrodes
    pos_ind = np.where(xcoor >= 1.5)[0]
    cse = data[:, 2:]  # first two columns are the coordinates

    return xcoor, cse, neg_ind, pos_ind


def run(time, dt, return_comsol=False):
    dtc = fem.Constant(dt)
    cmn, domain, comsol = common.prepare_comsol_buildup(time)
    pseudo_domain = cmn.pseudo_domain
    cse_domain = cmn.pseudo_cse_domain

    boundaries = np.arange(4)
    # Setup subdomain markers
    neg_domain = fem.CompiledSubDomain('(x[0] >= (b1 - DOLFIN_EPS)) && (x[0] <= (b2 + DOLFIN_EPS))',
                                       b1=boundaries[0].astype(np.double), b2=boundaries[1].astype(np.double))
    sep_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                       b1=boundaries[1].astype(np.double), b2=boundaries[2].astype(np.double))
    pos_domain = fem.CompiledSubDomain('(x[0] >= b1 - DOLFIN_EPS) && (x[0] <= b2 + DOLFIN_EPS)',
                                       b1=boundaries[2].astype(np.double), b2=boundaries[3].astype(np.double))

    combined_subdomains = fem.MeshFunction('size_t', domain.mesh, domain.mesh.topology().dim())
    combined_subdomains.array()[domain.domain_markers.array() == 0] = 1
    combined_subdomains.array()[domain.domain_markers.array() == 2] = 1
    submesh = fem.SubMesh(domain.mesh, combined_subdomains, 1)
    submesh_domain_markers = fem.MeshFunction('size_t', submesh, submesh.topology().dim())
    submesh_domain_markers.set_all(99)
    neg_domain.mark(submesh_domain_markers, 0)
    sep_domain.mark(submesh_domain_markers, 1)
    pos_domain.mark(submesh_domain_markers, 2)
    jV = fem.FunctionSpace(submesh, 'Lagrange', 1)

    # fem.plot(submesh)
    # plt.show()

    cs_sol = utilities.create_solution_matrices(int(len(time) / 2), len(comsol.pseudo_mesh), 1)[0]
    pseudo_cse_sol = \
        utilities.create_solution_matrices(int(len(time) / 2), len(cse_domain.mesh.coordinates()[:, 0]), 1)[0]
    cse_sol = utilities.create_solution_matrices(int(len(time) / 2), len(comsol.mesh), 1)[0]

    cs_u = fem.TrialFunction(pseudo_domain.V)
    v = fem.TestFunction(pseudo_domain.V)

    cs_1, cs, jbar2_interp = utilities.create_functions(pseudo_domain.V, 3)
    jbar_c, cse_2 = utilities.create_functions(domain.V, 2)
    cse_t = utilities.create_functions(jV, 1)[0]
    jbar2, cse = utilities.create_functions(cse_domain.V, 2)
    jbar2.set_allow_extrapolation(True)
    cse.set_allow_extrapolation(True)
    cse_t.set_allow_extrapolation(True)

    main_from_pseudo = fem.Expression(cppcode=x_conv, markers=cse_domain.domain_markers, degree=1)
    main_from_pseudo.neg = fem.Expression('x[0]', degree=1)
    main_from_pseudo.sep = fem.Expression('2*x[0]-1', degree=1)
    main_from_pseudo.pos = fem.Expression('x[0] + 0.5', degree=1)

    pseudo_from_main = fem.Expression(cppcode=x_conv, markers=submesh_domain_markers, degree=1)
    pseudo_from_main.neg = fem.Expression(('x[0]', '1.0'), degree=1)
    pseudo_from_main.sep = fem.Expression(('0.5*(x[0]+1)', '1.0'), degree=1)
    pseudo_from_main.pos = fem.Expression(('x[0] - 0.5', '1.0'), degree=1)

    composition_ex = fem.Expression(cppcode=composition, inner=main_from_pseudo, outer=jbar_c, degree=1)
    composition_ex_cse = fem.Expression(cppcode=composition, inner=pseudo_from_main, outer=cs, degree=1)

    Ds = cmn.fenics_params.Ds_ref
    Rs = cmn.fenics_params.Rs
    ds = pseudo_domain.ds
    dx = pseudo_domain.dx

    j, y, eps = sym.symbols('j x[1] DOLFIN_EPS')
    sym_j = sym.Piecewise((j, (y + eps) >= 1), (0, True))
    rbar2 = fem.Expression('pow(x[1], 2)', degree=1)
    jbar = fem.Expression(sym.printing.ccode(sym_j), j=jbar2, degree=3)
    neumann = dtc * rbar2 * (-jbar) * v * ds(5)
    a = Rs * rbar2 * cs_u * v * dx

    # Times at which to run solver
    time_in = [15, 25, 35, 45]

    # Collect common data
    V = pseudo_domain.V

    cs_1 = fem.Function(V)
    cs_1.vector()[:] = comsol.data.cs[-1, :].astype('double')

    # create local variables
    comsol_sol = cmn.comsol_solution

    k = 0
    for i in range(int(len(time) / 2)):
        i_1 = i * 2
        i = i*2 + 1

        utilities.assign_functions([comsol_sol.data.j], [jbar_c], domain.V, i_1)
        jbar2.assign(fem.interpolate(composition_ex, cse_domain.V))
        cs_1.vector()[:] = comsol.data.cs[i_1].astype('double')

        # Setup equation
        L = Rs*rbar2*cs_1*v*dx - dtc*Ds*rbar2/Rs*fem.dot(cs_1.dx(1), v.dx(1))*dx + neumann

        # Solve
        fem.solve(a == L, cs)
        cs.set_allow_extrapolation(True)
        cse.assign(fem.interpolate(cs, cse_domain.V))
        cse_t.assign(fem.interpolate(composition_ex_cse, jV))

        # plt.plot(utilities.get_1d(cse_t, jV)[0:21])
        # plt.plot(comsol.data.cse[i, comsol.neg_ind])
        # plt.show()
        # plt.plot(utilities.get_1d(cse_t, jV)[21:])
        # plt.plot(comsol.data.cse[i, comsol.pos_ind])
        # plt.show()

        pseudo_cse_sol[k, :] = cse.vector().get_local()
        cse_sol[k, :] = utilities.get_1d(fem.interpolate(cse_t, domain.V), domain.V)

        u_nodal_values = cs.vector()
        cs_sol[k, :] = u_nodal_values.get_local()[fem.vertex_to_dof_map(pseudo_domain.V)]
        k += 1

    # exit()
    ## find c_se from c_s
    xcoor, cse, neg_ind, pos_ind = find_cse_from_cs(comsol)
    utilities.report(xcoor[neg_ind], time_in, pseudo_cse_sol[:, neg_ind],
                     cse.T[:, neg_ind], '$c_{s,e}^{neg}$')
    plt.show()
    utilities.report(xcoor[pos_ind], time_in, pseudo_cse_sol[:, pos_ind],
                     cse.T[:, pos_ind], '$c_{s,e}^{pos}$')
    plt.show()
    # plt.savefig('comsol_compare_cs.png')
    # plt.show()

    if return_comsol:
        return cse_sol, comsol
    else:
        return cse_sol


def main():
    # Quiet
    fem.set_log_level(fem.ERROR)

    # Times at which to run solver
    time_in = [15, 25, 35, 45]
    # time_in = np.arange(0.1, 50, 0.1)
    dt = 0.1
    time = [None] * (len(time_in) * 2)
    time[::2] = [t - dt for t in time_in]
    time[1::2] = time_in

    cse_sol, comsol = run(time, dt, return_comsol=True)

    utilities.report(comsol.mesh[comsol.neg_ind], time_in, cse_sol[:, comsol.neg_ind],
                     comsol.data.cse[1::2, comsol.neg_ind], '$\c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_neg.png')
    plt.show()
    utilities.report(comsol.mesh[comsol.pos_ind], time_in, cse_sol[:, comsol.pos_ind],
                     comsol.data.cse[1::2, comsol.pos_ind], '$\c_{s,e}$')
    utilities.save_plot(__file__, 'plots/compare_cse_pos.png')

    plt.show()


if __name__ == '__main__':
    main()
