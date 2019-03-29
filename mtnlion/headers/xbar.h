#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/geometry/BoundingBoxTree.h>

#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>

class XBar : public dolfin::Expression {
   public:
    XBar() : dolfin::Expression(), allow_extrapolation(false) {}

    ufc::cell calc_cell(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
         if (space == nullptr){
            dolfin::dolfin_error("xbar.h", "evaluate expression without cell reference", "No function space defined. Don't forget to bind space");
        }

        // Find the cell that contains x
        const dolfin::Mesh& mesh = *space->mesh();
        const double* _x = x.data();

        // Get index of first cell containing point
        const dolfin::Point point(mesh.geometry().dim(), _x);
        unsigned int id = mesh.bounding_box_tree()->compute_first_entity_collision(point);

        // If not found, use the closest cell
  if (id == std::numeric_limits<unsigned int>::max())
  {
    // Check if the closest cell is within DOLFIN_EPS. This we can
    // allow without allow_extrapolation
    std::pair<unsigned int, double> close
      = mesh.bounding_box_tree()->compute_closest_entity(point);

    if (allow_extrapolation or close.second < DOLFIN_EPS)
      id = close.first;
    else
    {
      dolfin::dolfin_error("xbar.cpp",
                   "evaluate function at point",
                   "The point is not inside the domain. Consider calling \"Function::setallow_extrapolation(true)\" on this Function to allow extrapolation");
    }
  }

  // Create cell that contains point
  const dolfin::Cell cell(mesh, id);
  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);
  return ufc_cell;
    }

    void do_eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {
        switch ((*markers)[cell.index]) {
            case 0:
                py::print("\tXBar 1 neg");
                neg->eval(values, x);
                break;
            case 1:
                py::print("\tXBar 3 sep");
                sep->eval(values, x);
                break;
            case 2:
                py::print("\tXBar 5 pos");
                pos->eval(values, x);
                break;
        }
    }

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {
        ufc::cell ufc_cell = cell;

        if(space != nullptr) {
            dolfin::Array<double> _values(values.size(), values.data());
            const dolfin::Array<double> _x(x.size(), const_cast<double*>(x.data()));

            ufc_cell = calc_cell(_values, _x);
        }

        py::print("XBar size: ", markers->size(), " index: ", ufc_cell.index);
        py::print("\tOld x: ", x[0]);
//        unsigned index = cell.index < markers->size() ? cell.index : markers->size()-1;
        do_eval(values, x, ufc_cell);
        py::print("\tNew x: ", values[0]);
    }

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const {
        dolfin::Array<double> _values(values.size(), values.data());
        const dolfin::Array<double> _x(x.size(), const_cast<double*>(x.data()));

        ufc::cell ufc_cell = calc_cell(_values, _x);

        // Call evaluate function
        do_eval(values, x, ufc_cell);
    }

    std::shared_ptr<dolfin::MeshFunction<std::size_t>> markers;
    std::shared_ptr<const dolfin::FunctionSpace> space;
    std::shared_ptr<dolfin::GenericFunction> neg;
    std::shared_ptr<dolfin::GenericFunction> sep;
    std::shared_ptr<dolfin::GenericFunction> pos;
    bool allow_extrapolation;
};

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<XBar, std::shared_ptr<XBar>, dolfin::Expression>(m, "XBar")
        .def(py::init<>())
        .def_readwrite("markers", &XBar::markers)
        .def_readwrite("neg", &XBar::neg)
        .def_readwrite("sep", &XBar::sep)
        .def_readwrite("pos", &XBar::pos)
        .def_readwrite("space", &XBar::space)
        .def_readwrite("allow_extrapolation", &XBar::allow_extrapolation);
}
