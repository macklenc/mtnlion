#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/mesh/MeshFunction.h>

class Piecewise : public dolfin::Expression {
   public:
    Piecewise() : dolfin::Expression() {}

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {
        switch ((*markers)[cell.index]) {
            case 0:
                k_1->eval(values, x);
                break;
            case 1:
                k_2->eval(values, x);
                break;
            case 2:
                k_3->eval(values, x);
                break;
            case 3:
                k_4->eval(values, x);
                break;
        }
    }

    std::shared_ptr<dolfin::MeshFunction<std::size_t>> markers;
    std::shared_ptr<dolfin::GenericFunction> k_1;
    std::shared_ptr<dolfin::GenericFunction> k_2;
    std::shared_ptr<dolfin::GenericFunction> k_3;
    std::shared_ptr<dolfin::GenericFunction> k_4;
};

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<Piecewise, std::shared_ptr<Piecewise>, dolfin::Expression>(m, "Piecewise")
        .def(py::init<>())
        .def_readwrite("markers", &Piecewise::markers)
        .def_readwrite("k_1", &Piecewise::k_1)
        .def_readwrite("k_2", &Piecewise::k_2)
        .def_readwrite("k_3", &Piecewise::k_3)
        .def_readwrite("k_4", &Piecewise::k_4);
}
