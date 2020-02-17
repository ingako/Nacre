#include <pybind11/pybind11.h>
#include "pro_pearl.h"

namespace py = pybind11;

PYBIND11_MODULE(pearl, m) {
    m.doc() = "PRO_PEARL's implementation in C++";

    py::class_<pro_pearl, pearl>(m, "pro_pearl")
        .def(py::init<int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      double,
                      double,
                      double,
                      double,
                      double >())
        .def_property_readonly("drift_detected", &pro_pearl::get_drift_detected)
        .def("find_last_actual_drift_point", &pro_pearl::find_last_actual_drift_point)
        .def("select_candidate_trees_proactively", &pro_pearl::select_candidate_trees_proactively)
        .def("adapt_state_proactively", &pro_pearl::adapt_state_proactively)
        .def("process", &pro_pearl::process)
        .def("adapt_state", &pro_pearl::adapt_state);
}
