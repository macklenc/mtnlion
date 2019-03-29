#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>

class {CLASS_NAME} : public dolfin::Expression {{
   public:
    {CLASS_NAME}() : dolfin::Expression() {{}}

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {{
        {COMMANDS}
    }}

    {GENERIC_FUNCTIONS}
}};

PYBIND11_MODULE(SIGNATURE, m) {{
    py::class_<{CLASS_NAME}, std::shared_ptr<{CLASS_NAME}>, dolfin::Expression>(m, "{CLASS_NAME}")
        .def(py::init<>()){EXPOSE_GENERIC_FUNCTIONS};
}}
