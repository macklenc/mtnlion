import numbers

import fenics as fem
import munch
import numpy as np
import sympy as sym

import domain2
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import utilities


def prepare_comsol_buildup(time):
    cmn = Common(time)
    domain = cmn.domain
    comsol = cmn.comsol_solution

    return cmn, domain, comsol

class Domain():
    def __init__(self, V, dx, ds, dS, n, boundary_markers, domain_markers):
        self.V = V
        self.dx = dx
        self.ds = ds
        self.dS = dS
        self.n = n
        self.neg_marker, self.sep_marker, self.pos_marker = (0, 1, 2)
        self.boundary_markers = boundary_markers
        self.domain_markers = domain_markers


def collect_params(params, mesh, dm):
    lst = [0, 0]
    n_dict = dict()
    for k in params.neg:
        neg = params.neg.get(k)
        sep = params.sep.get(k, None)
        pos = params.pos.get(k, None)

        if sep is None:
            sep = 0
        else:
            lst[0] += 1

        if pos is None:
            pos = 0
        else:
            lst[1] += 1

        if isinstance(neg, numbers.Number):
            n_dict[k] = utilities.piecewise(mesh, dm, neg, sep, pos)
        else:
            n_dict[k] = [neg, sep, pos]

    if len(params.sep) < lst[0] or len(params.pos) < lst[1]:
        raise NameError('Not all parameters were added')

    return munch.Munch(n_dict)


def collect_const(const):
    n_dict = dict()
    for k, v in const.items():
        if isinstance(v, numbers.Number):
            n_dict[k] = fem.Constant(v)
        else:
            n_dict[k] = v

    return munch.Munch(n_dict)


def collect(n_dict, *keys):
    return [n_dict[x] for x in keys]


def kappa_D(R, Tref, F, t_plus, **kwargs):
    dfdc = sym.Symbol('dfdc')
    # dfdc = 0
    kd = fem.Constant(2) * R * Tref / F * (fem.Constant(1) + dfdc) * (t_plus - fem.Constant(1))
    kappa_D = fem.Expression(sym.printing.ccode(kd), dfdc=0, degree=1)

    return kd, kappa_D


def kappa_Deff(ce, kappa_ref, eps_e, brug_kappa, kappa_D, **kwargs):
    x = sym.Symbol('ce')
    y = sym.Symbol('x')
    kp = kappa_ref.subs(y, x)

    # TODO: separate kappa_ref
    kappa_ref = fem.Expression(sym.printing.ccode(kp), ce=ce, degree=1)
    kappa_eff = kappa_ref * eps_e ** brug_kappa
    kappa_Deff = kappa_D * kappa_ref * eps_e

    return kappa_eff, kappa_Deff


class Common:
    def __init__(self, time):
        self.time = time

        # Collect required data
        comsol_data, raw_params = utilities.gather_data()
        self.time_ind = engine.find_ind_near(comsol_data.time_mesh, time)
        self.comsol_solution = comsol.get_standardized(comsol_data.filter_time(self.time_ind))
        self.comsol_solution.data.cse[np.isnan(self.comsol_solution.data.cse)] = 0
        self.comsol_solution.data.phis[np.isnan(self.comsol_solution.data.phis)] = 0

        self.mesh, self.dx, self.ds, self.dS, self.n, self.bm, self.dm = \
            domain2.generate_domain(self.comsol_solution.mesh)
        # self.neg_submesh = fem.SubMesh(self.mesh, self.dm, 0)
        # self.sep_submesh = fem.SubMesh(self.mesh, self.dm, 1)
        # self.pos_submesh = fem.SubMesh(self.mesh, self.dm, 2)

        # Initialize parameters
        self.const = collect_const(raw_params.const)
        self.const['F'] = fem.Constant(96487)
        self.const['R'] = fem.Constant(8.314)  # universal gas constant

        self.params = collect_params(raw_params, self.mesh, self.dm)
        self.params['a_s'] = fem.Constant(3) * utilities.piecewise(self.mesh, self.dm,
                                                                   raw_params.neg.eps_s / raw_params.neg.Rs, 0,
                                                                   raw_params.pos.eps_s / raw_params.pos.Rs)
        self.params['sigma_eff'] = self.params.sigma_ref * self.params.eps_s ** self.params.brug_sigma
        self.params['De_eff'] = self.const.De_ref * self.params.eps_e ** self.params.brug_De
        self.params['Uocp_neg'] = self.params.Uocp[0][0]
        self.params['Uocp_pos'] = self.params.Uocp[2][0]
        self.const.kappa_ref = self.const.kappa_ref[0]

        self.V = fem.FunctionSpace(self.mesh, 'Lagrange', 1)
        # self.neg_V = fem.FunctionSpace(self.neg_submesh, 'Lagrange', 1)
        # self.sep_V = fem.FunctionSpace(self.sep_submesh, 'Lagrange', 1)
        # self.pos_V = fem.FunctionSpace(self.pos_submesh, 'Lagrange', 1)
        self.domain = Domain(self.V, self.dx, self.ds, self.dS, self.n, self.bm, self.dm)

        self.I_1C = 20.5
        self.Iapp = [self.I_1C if 10 <= i <= 20 else -self.I_1C if 30 <= i <= 40 else 0 for i in time]

        _, self.params['kappa_D'] = kappa_D(**self.params, **self.const)


class Common2:
    def __init__(self, time, mesh):
        self.time = time

        # Collect required data
        self.comsol_data, self.params = utilities.gather_data()
        self.time_ind = engine.find_ind_near(self.comsol_data.time_mesh, time)
        self.comsol_solution = comsol.get_standardized(self.comsol_data.filter_time(self.time_ind))
        self.comsol_solution.data.cse[np.isnan(self.comsol_solution.data.cse)] = 0
        self.comsol_solution.data.phis[np.isnan(self.comsol_solution.data.phis)] = 0

        self.mesh, self.dx, self.ds, self.bm, self.dm = domain2.generate_domain2(mesh)

        # Initialize parameters
        self.F = fem.Constant(96487)
        self.R = fem.Constant(8.314)  # universal gas constant
        self.T = fem.Constant(298.15)
        self.I_1C = fem.Constant(20.5)
        self.Iapp = [self.I_1C if 10 <= i <= 20 else -self.I_1C if 30 <= i <= 40 else fem.Constant(0) for i in time]
        self.Acell = fem.Constant(self.params.const.Acell)

        self.Lc = utilities.mkparam(self.dm, self.params.neg.L, self.params.sep.L, self.params.pos.L)
        self.sigma_ref = utilities.mkparam(self.dm, self.params.neg.sigma_ref, 0, self.params.pos.sigma_ref)
        self.eps_s = utilities.mkparam(self.dm, self.params.neg.eps_s, 0, self.params.pos.eps_s)
        self.brug_sigma = utilities.mkparam(self.dm, self.params.neg.brug_sigma, 0, self.params.pos.brug_sigma)
        self.sigma_eff = self.sigma_ref * self.eps_s ** self.brug_sigma
        self.Rs = utilities.mkparam(self.dm, self.params.neg.Rs, 0, self.params.pos.Rs)
        self.a_s = fem.Constant(3) * utilities.mkparam(self.dm, self.params.neg.eps_s / self.params.neg.Rs, 0,
                                                       self.params.pos.eps_s / self.params.pos.Rs)

        self.eps_e = utilities.mkparam(self.dm, self.params.neg.eps_e, self.params.sep.eps_e, self.params.pos.eps_e)
        self.brug_De = utilities.mkparam(self.dm, self.params.neg.brug_De,
                                         self.params.sep.brug_De, self.params.pos.brug_De)
        self.De_ref = fem.Constant(self.params.const.De_ref)
        self.de_eff = self.De_ref * self.eps_e ** self.brug_De
        self.t_plus = fem.Constant(self.params.const.t_plus)

        self.k_norm_ref = utilities.mkparam(self.dm, self.params.neg.k_norm_ref, 0, self.params.pos.k_norm_ref)
        self.csmax = utilities.mkparam(self.dm, self.params.neg.csmax, 0, self.params.pos.csmax)
        self.alpha = utilities.mkparam(self.dm, self.params.neg.alpha, 0, self.params.pos.alpha)
        self.ce0 = fem.Constant(self.params.const.ce0)
        self.Tref = fem.Constant(self.params.const.Tref)

        # self.cs0 = utilities.mkparam(self.dm, self.params.csmax*)
        self.brug_kappa = utilities.mkparam(self.dm, self.params.neg.brug_kappa, self.params.sep.brug_kappa,
                                            self.params.pos.brug_kappa)



        self.Ds = utilities.mkparam(self.dm, self.params.neg.Ds_ref, 0, self.params.pos.Ds_ref)

# Notes for later, TODO: clean up
#         boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
#         boundary_markers.set_all(0)
#         b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=1)
#         b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=2)
#         b1.mark(boundary_markers, 2)
#         b2.mark(boundary_markers, 3)