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


def collect_fenics_const(const):
    n_dict = dict()
    for k, v in const.items():
        if isinstance(v, numbers.Number):
            n_dict[k] = fem.Constant(v)
        else:
            n_dict[k] = v

    return munch.Munch(n_dict)


def collect_fenics_params(params, mesh, dm, V):
    n_dict = dict()

    for k, v in params.items():
        if isinstance(v, np.ndarray):
            try:
                n_dict[k] = utilities.piecewise(mesh, dm, V, *v)
            except TypeError:
                n_dict[k] = v
        else:
            n_dict[k] = v

    return munch.Munch(n_dict)


def collect(n_dict, *keys):
    return [n_dict[x] for x in keys]


def kappa_D(R, Tref, F, t_plus, dfdc, **kwargs):
    kd = 2 * R * Tref / F * (1 + dfdc) * (t_plus - 1)

    return kd


def kappa_Deff(ce, kappa_ref, eps_e, brug_kappa, kappa_D, **kwargs):
    x = sym.Symbol('ce')
    y = sym.Symbol('x')
    kp = kappa_ref.subs(y, x)

    # TODO: separate kappa_ref
    kappa_ref = fem.Expression(sym.printing.ccode(kp), ce=ce, degree=1)
    kappa_eff = kappa_ref * eps_e ** brug_kappa
    kappa_Deff = kappa_D * kappa_ref * eps_e

    return kappa_eff, kappa_Deff


def collect_parameters(params):
    n_dict = dict()

    keys = set(list(params.neg.keys()) + list(params.sep.keys()) + list(params.pos.keys()))
    for k in keys:
        neg = params.neg.get(k, 0.0)
        sep = params.sep.get(k, 0.0)
        pos = params.pos.get(k, 0.0)

        n_dict[k] = np.array([neg, sep, pos])

    return munch.Munch(n_dict)


# TODO: settable tolerances
# TODO: Documentation
def organize(file_coords, dofs):
    transform = []
    for i in dofs:
        ind1 = np.where(np.abs(file_coords[:, 0] - i[0]) <= 1e-5)
        ind2 = np.where(np.abs(file_coords[:, 1] - i[1]) <= 1e-5)
        if len(ind1[0]) > 0 and len(ind2[0]) > 0:
            transform.append(np.intersect1d(ind1, ind2)[0])
            if len(np.intersect1d(ind1, ind2)) > 1:
                raise ValueError('Too many matching indices')
        else:
            raise ValueError('Missing indices, check tolerances')
    return transform


# TODO: Documentation
def get_fenics_dofs(mesh_xml):
    mesh = fem.Mesh(mesh_xml)
    V = fem.FunctionSpace(mesh, 'Lagrange', 1)
    dofs = V.tabulate_dof_coordinates().reshape(-1, 2)
    return dofs


class Common:
    def __init__(self, time):
        self.time = time

        # Collect required data
        comsol_data, self.raw_params, pseudo_mesh_file = utilities.gather_data()

        self.time_ind = engine.find_ind_near(comsol_data.time_mesh, time)
        self.comsol_solution = comsol.get_standardized(comsol_data.filter_time(self.time_ind))
        self.comsol_solution.data.cse[np.isnan(self.comsol_solution.data.cse)] = 0
        self.comsol_solution.data.phis[np.isnan(self.comsol_solution.data.phis)] = 0

        pseudo_mesh_dofs = get_fenics_dofs(pseudo_mesh_file)
        shuffle_indices = organize(self.comsol_solution.pseudo_mesh, pseudo_mesh_dofs)
        self.comsol_solution.pseudo_mesh = self.comsol_solution.pseudo_mesh[shuffle_indices]
        self.comsol_solution.data.cs = self.comsol_solution.data.cs[:, shuffle_indices]


        self.params = collect_parameters(self.raw_params)
        self.consts = self.raw_params.const

        self.params['Uocp_neg'] = self.params.Uocp[0][0]
        self.params['Uocp_pos'] = self.params.Uocp[2][0]
        self.params['De_eff'] = self.consts.De_ref * self.params.eps_e ** self.params.brug_De
        self.params['sigma_eff'] = self.params.sigma_ref * self.params.eps_s ** self.params.brug_sigma
        self.params['a_s'] = 3 * np.divide(self.params.eps_s, self.params.Rs,
                                           out=np.zeros_like(self.params.eps_s), where=self.params.Rs != 0)

        self.consts['F'] = 96487
        self.consts['R'] = 8.314
        self.consts['dfdc'] = 0
        self.consts['kappa_D'] = kappa_D(**self.params, **self.consts)
        self.consts.kappa_ref = self.consts.kappa_ref[0]

        self.mesh, self.dx, self.ds, self.dS, self.n, self.bm, self.dm = \
            domain2.generate_domain(self.comsol_solution.mesh)

        self.V = fem.FunctionSpace(self.mesh, 'Lagrange', 1)
        self.V0 = fem.FunctionSpace(self.mesh, 'DG', 0)

        self.fenics_params = collect_fenics_params(self.params, self.mesh, self.dm, self.V0)
        self.fenics_consts = collect_fenics_const(self.consts)

        # self.neg_submesh = fem.SubMesh(self.mesh, self.dm, 0)
        # self.sep_submesh = fem.SubMesh(self.mesh, self.dm, 1)
        # self.pos_submesh = fem.SubMesh(self.mesh, self.dm, 2)

        # self.neg_V = fem.FunctionSpace(self.neg_submesh, 'Lagrange', 1)
        # self.sep_V = fem.FunctionSpace(self.sep_submesh, 'Lagrange', 1)
        # self.pos_V = fem.FunctionSpace(self.pos_submesh, 'Lagrange', 1)
        self.domain = Domain(self.V, self.dx, self.ds, self.dS, self.n, self.bm, self.dm)

        self.I_1C = 20.5
        self.Iapp = [self.I_1C if 10 <= i <= 20 else -self.I_1C if 30 <= i <= 40 else 0 for i in time]


class Common2:
    def __init__(self, time, mesh):
        self.time = time

        # Collect required data
        self.comsol_data, self.params, _ = utilities.gather_data()
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
