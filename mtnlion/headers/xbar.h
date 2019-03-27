#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/mesh/MeshFunction.h>

class XBar : public dolfin::Expression
{
public:
	XBar() : dolfin::Expression() {}

	void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {
		switch((*markers)[cell.index]){
			case 0:
				neg->eval(values, x);
				break;
			case 1:
				sep->eval(values, x);
				break;
			case 2:
				pos->eval(values, x);
				break;
		}
	}

	std::shared_ptr<dolfin::MeshFunction<std::size_t>> markers;
	std::shared_ptr<dolfin::GenericFunction> neg;
	std::shared_ptr<dolfin::GenericFunction> sep;
	std::shared_ptr<dolfin::GenericFunction> pos;
};

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<XBar, std::shared_ptr<XBar>, dolfin::Expression>
    (m, "XBar")
    .def(py::init<>())
    .def_readwrite("markers", &XBar::markers)
    .def_readwrite("neg", &XBar::neg)
    .def_readwrite("sep", &XBar::sep)
    .def_readwrite("pos", &XBar::pos);
}
