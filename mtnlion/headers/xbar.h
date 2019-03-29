#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/mesh/MeshFunction.h>

class XBar : public dolfin::Expression {
   public:
    XBar() : dolfin::Expression() {}

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {
        py::print("XBar 0 ", cell.index, markers->size());
        py::print(markers->mesh()->num_cells());
        py::print("Old x: ", x[0]);
        unsigned index = cell.index < markers->size() ? cell.index : markers->size()-1;
        switch ((*markers)[index]) {
            case 0:
                py::print("XBar 1 neg");
                neg->eval(values, x);
                py::print("XBar 2");
                break;
            case 1:
                py::print("XBar 3 sep");
                sep->eval(values, x);
                py::print("XBar 4");
                break;
            case 2:
                py::print("XBar 5 pos");
                pos->eval(values, x);
                py::print("XBar 6");
                break;
        }
        py::print("New x: ", values[0]);
    }

    std::shared_ptr<dolfin::MeshFunction<std::size_t>> markers;
    std::shared_ptr<dolfin::GenericFunction> neg;
    std::shared_ptr<dolfin::GenericFunction> sep;
    std::shared_ptr<dolfin::GenericFunction> pos;
};

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<XBar, std::shared_ptr<XBar>, dolfin::Expression>(m, "XBar")
        .def(py::init<>())
        .def_readwrite("markers", &XBar::markers)
        .def_readwrite("neg", &XBar::neg)
        .def_readwrite("sep", &XBar::sep)
        .def_readwrite("pos", &XBar::pos);
}
