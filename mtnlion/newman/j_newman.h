#include <dolfin/function/Expression.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>

class J_Newman : public dolfin::Expression {
   public:
    J_Newman() : dolfin::Expression() {}

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& c) const override {
        double Uocp, Tref, R, F, k_norm_ref, alpha, ce0, csmax, phis, phie, cse, ce;
        generic_function_Uocp->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&Uocp), x, c);
        generic_function_Tref->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&Tref), x, c);
        generic_function_R->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&R), x, c);
        generic_function_F->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&F), x, c);
        generic_function_k_norm_ref->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&k_norm_ref), x, c);

        generic_function_alpha->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&alpha), x, c);
        generic_function_ce0->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&ce0), x, c);
        generic_function_csmax->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&csmax), x, c);
        generic_function_phis->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&phis), x, c);
        generic_function_phie->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&phie), x, c);
        generic_function_cse->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&cse), x, c);
        generic_function_ce->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&ce), x, c);
        values[0] =
            ((x[0] >= 2 || x[0] <= 1)
                 ? (k_norm_ref *
                    pow(((1.0L / 2.0L) * (((cse / csmax) > 0) - ((cse / csmax) < 0)) + 1.0L / 2.0L) * fabs(cse / csmax),
                        alpha) *
                    pow(((1.0L / 2.0L) * (((ce * (-cse + csmax) / (ce0 * csmax)) > 0) -
                                          ((ce * (-cse + csmax) / (ce0 * csmax)) < 0)) +
                         1.0L / 2.0L) *
                            fabs(ce * (cse - csmax) / (ce0 * csmax)),
                        -alpha + 1) *
                    (exp(F * (-alpha + 1) * (-Uocp - phie + phis) / (R * Tref)) -
                     exp(-F * alpha * (-Uocp - phie + phis) / (R * Tref))))
                 : (0));
    }
    std::shared_ptr<dolfin::GenericFunction> generic_function_ce;
    std::shared_ptr<dolfin::GenericFunction> generic_function_cse;
    std::shared_ptr<dolfin::GenericFunction> generic_function_phie;
    std::shared_ptr<dolfin::GenericFunction> generic_function_phis;
    std::shared_ptr<dolfin::GenericFunction> generic_function_csmax;
    std::shared_ptr<dolfin::GenericFunction> generic_function_ce0;
    std::shared_ptr<dolfin::GenericFunction> generic_function_alpha;
    std::shared_ptr<dolfin::GenericFunction> generic_function_k_norm_ref;
    std::shared_ptr<dolfin::GenericFunction> generic_function_F;
    std::shared_ptr<dolfin::GenericFunction> generic_function_R;
    std::shared_ptr<dolfin::GenericFunction> generic_function_Tref;
    std::shared_ptr<dolfin::GenericFunction> generic_function_Uocp;
};

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<J_Newman, std::shared_ptr<J_Newman>, dolfin::Expression>(m, "J_Newman")
        .def(py::init<>())
        .def_readwrite("ce", &J_Newman::generic_function_ce)
        .def_readwrite("cse", &J_Newman::generic_function_cse)
        .def_readwrite("phie", &J_Newman::generic_function_phie)
        .def_readwrite("phis", &J_Newman::generic_function_phis)
        .def_readwrite("csmax", &J_Newman::generic_function_csmax)
        .def_readwrite("ce0", &J_Newman::generic_function_ce0)
        .def_readwrite("alpha", &J_Newman::generic_function_alpha)
        .def_readwrite("k_norm_ref", &J_Newman::generic_function_k_norm_ref)
        .def_readwrite("F", &J_Newman::generic_function_F)
        .def_readwrite("R", &J_Newman::generic_function_R)
        .def_readwrite("Tref", &J_Newman::generic_function_Tref)
        .def_readwrite("Uocp", &J_Newman::generic_function_Uocp);
}

