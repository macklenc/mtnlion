#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/geometry/BoundingBoxTree.h>

#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>


// TODO: 1) Cleanup 2) refactor calc_cell to use Eigen

class XBar : public dolfin::Expression {
   public:
    XBar() : dolfin::Expression(), allow_extrapolation(false) {}

    ufc::cell calc_cell(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
         if (mesh == nullptr){
            dolfin::dolfin_error("xbar.h", "evaluate expression without cell reference", "No mesh defined. Don't forget to bind a mesh");
        }

        // Find the cell that contains x
        const dolfin::Mesh& _mesh = *mesh;
        const double* _x = x.data();

        // Get index of first cell containing point
        const dolfin::Point point(_mesh.geometry().dim(), _x);
        unsigned int id = _mesh.bounding_box_tree()->compute_first_entity_collision(point);

        // If not found, use the closest cell
  if (id == std::numeric_limits<unsigned int>::max())
  {
    // Check if the closest cell is within DOLFIN_EPS. This we can
    // allow without allow_extrapolation
    std::pair<unsigned int, double> close
      = _mesh.bounding_box_tree()->compute_closest_entity(point);

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
  const dolfin::Cell cell(_mesh, id);
  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);
  return ufc_cell;
    }

    void do_eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {
        switch ((*markers)[cell.index]) {
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

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const {
        ufc::cell ufc_cell = cell;

        if(mesh != nullptr) {
            dolfin::Array<double> _values(values.size(), values.data());
            const dolfin::Array<double> _x(x.size(), const_cast<double*>(x.data()));

            ufc_cell = calc_cell(_values, _x);
        }

//        unsigned index = cell.index < markers->size() ? cell.index : markers->size()-1;
        do_eval(values, x, ufc_cell);
    }

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const {
        dolfin::Array<double> _values(values.size(), values.data());
        const dolfin::Array<double> _x(x.size(), const_cast<double*>(x.data()));

        ufc::cell ufc_cell = calc_cell(_values, _x);

        // Call evaluate function
        do_eval(values, x, ufc_cell);
    }

    std::shared_ptr<dolfin::MeshFunction<std::size_t>> markers;
    std::shared_ptr<const dolfin::Mesh> mesh;
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
        .def_readwrite("mesh", &XBar::mesh)
        .def_readwrite("allow_extrapolation", &XBar::allow_extrapolation);
}
