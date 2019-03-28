#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>

class Composition : public dolfin::Expression
{
public:
    Composition() : dolfin::Expression() {}

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const
    {
        Eigen::VectorXd val(3);
        inner->eval(val, x, c);
        outer->eval(values, val, c);
    }

//  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
//  {
//      Eigen::VectorXd val(3);
//      inner->eval(val, x);
//      outer->eval(values, val);
//  }

    std::shared_ptr<dolfin::GenericFunction> outer;
    std::shared_ptr<dolfin::GenericFunction> inner;
};

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<Composition, std::shared_ptr<Composition>, dolfin::Expression>
    (m, "Composition")
    .def(py::init<>())
    .def_readwrite("outer", &Composition::outer)
    .def_readwrite("inner", &Composition::inner);
}
