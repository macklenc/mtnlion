import numbers

import fenics as fem
import munch
import numpy as np
import sympy as sym
from scipy import interpolate as intp

import domain2
import mtnlion.comsol as comsol
import mtnlion.engine as engine
import utilities


def prepare_comsol_buildup():
    # cmn = Common(time)
    cmn = Common()
    domain = cmn.domain
    comsol = cmn.comsol_solution

    return cmn, domain, comsol


def collect_fenics_const(const):
    n_dict = dict()
    for k, v in const.items():
        if isinstance(v, numbers.Number):
            n_dict[k] = fem.Constant(v)
        else:
            n_dict[k] = v

    return munch.Munch(n_dict)


def collect_fenics_params(params, func):
    n_dict = dict()

    for k, v in params.items():
        if isinstance(v, np.ndarray):
            n_dict[k] = func(v)
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
    def __init__(self):
        # self.time = time

        # Collect required data
        comsol_data, self.raw_params, pseudo_mesh_file, Uocp_spline, input_current = utilities.gather_data()

        self.time = comsol_data.time_mesh
        self.time_ind = engine.find_ind_near(comsol_data.time_mesh, self.time)
        self.comsol_solution = comsol.get_standardized(comsol_data.filter_time(self.time_ind))
        self.comsol_solution.data.cse[np.isnan(self.comsol_solution.data.cse)] = 0
        self.comsol_solution.data.phis[np.isnan(self.comsol_solution.data.phis)] = 0

        pseudo_mesh_dofs = get_fenics_dofs(pseudo_mesh_file)
        shuffle_indices = organize(self.comsol_solution.pseudo_mesh, pseudo_mesh_dofs)
        self.comsol_solution.pseudo_mesh = self.comsol_solution.pseudo_mesh[shuffle_indices]
        self.comsol_solution.data.cs = self.comsol_solution.data.cs[:, shuffle_indices]

        self.Uocp_spline = munch.Munch(Uocp_spline)

        self.params = collect_parameters(self.raw_params)
        self.consts = self.raw_params.const

        tmpx, soc = sym.symbols('x soc')
        self.params['Uocp_neg'] = self.params.Uocp[0][0].subs(tmpx, soc)
        self.params['Uocp_pos'] = self.params.Uocp[2][0].subs(tmpx, soc)
        self.params['De_eff'] = self.consts.De_ref * self.params.eps_e ** self.params.brug_De
        self.params['sigma_eff'] = self.params.sigma_ref * self.params.eps_s ** self.params.brug_sigma
        self.params['a_s'] = 3 * np.divide(self.params.eps_s, self.params.Rs,
                                           out=np.zeros_like(self.params.eps_s), where=self.params.Rs != 0)
        self.params['cs_0'] = [self.params.csmax[0] * 0.325, 0, self.params.csmax[2] * 0.422]

        self.consts['F'] = 96487
        self.consts['R'] = 8.314
        self.consts['dfdc'] = 0
        self.consts['kappa_D'] = kappa_D(**self.params, **self.consts)
        self.consts.kappa_ref = self.consts.kappa_ref[0]

        self.domain, self.pseudo_domain, self.pseudo_cse_domain, self.electrode_domain = \
            domain2.generate_domain(self.comsol_solution.mesh, fem.Mesh(pseudo_mesh_file))

        self.V0 = fem.FunctionSpace(self.domain.mesh, 'DG', 0)
        self.V0_electrode = fem.FunctionSpace(self.electrode_domain.mesh, 'DG', 0)

        def fenics_params_func(v):
            try:
                if v[1] != 0:
                    return utilities.piecewise(self.domain.mesh, self.domain.domain_markers, self.V0, *v)
                else:
                    return utilities.piecewise(self.electrode_domain.mesh, self.electrode_domain.domain_markers,
                                               self.V0_electrode, *v)
            except TypeError:
                return v

        self.fenics_params = collect_fenics_params(self.params, fenics_params_func)
        x = sym.Symbol('x[0]')
        self.fenics_params['uocp_str'] = sym.Piecewise((self.params.Uocp_neg, x <= 1 + fem.DOLFIN_EPS),
                                                       (self.params.Uocp_pos, x >= 2 - fem.DOLFIN_EPS),
                                                       (0, True))
        self.fenics_params.pop('Uocp')

        self.fenics_consts = collect_fenics_const(self.consts)

        # TODO: refactor
        Rs = utilities.mkparam(self.pseudo_domain.domain_markers, fem.Constant(self.params.Rs[0]),
                               fem.Constant(self.params.Rs[1]), fem.Constant(self.params.Rs[2]))
        Ds = utilities.mkparam(self.pseudo_domain.domain_markers, fem.Constant(self.params.Ds_ref[0]),
                               fem.Constant(self.params.Ds_ref[1]), fem.Constant(self.params.Ds_ref[2]))
        cs_0 = utilities.mkparam(self.pseudo_domain.domain_markers, fem.Constant(self.params.cs_0[0]),
                                 fem.Constant(self.params.cs_0[1]), fem.Constant(self.params.cs_0[2]))
        self.fenics_params.Rs = Rs
        self.fenics_params.Ds_ref = Ds
        self.fenics_params.cs_0 = cs_0

        # self.neg_submesh = fem.SubMesh(self.mesh, self.dm, 0)
        # self.sep_submesh = fem.SubMesh(self.mesh, self.dm, 1)
        # self.pos_submesh = fem.SubMesh(self.mesh, self.dm, 2)

        # self.neg_V = fem.FunctionSpace(self.neg_submesh, 'Lagrange', 1)
        # self.sep_V = fem.FunctionSpace(self.sep_submesh, 'Lagrange', 1)
        # self.pos_V = fem.FunctionSpace(self.pos_submesh, 'Lagrange', 1)

        # self.I_1C = 20.5
        # self.Iapp = [self.I_1C if 10 <= i < 20 else -self.I_1C if 30 <= i < 40 else 0 for i in time]
        self.Iapp = intp.interp1d(input_current[:, 0], input_current[:, 1], kind='cubic')


# Notes for later, TODO: clean up
#         boundary_markers = fem.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
#         boundary_markers.set_all(0)
#         b1 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=1)
#         b2 = fem.CompiledSubDomain('near(x[0], b, DOLFIN_EPS)', b=2)
#         b1.mark(boundary_markers, 2)
#         b2.mark(boundary_markers, 3)
