#include <pybind11/pybind11.h>

#include <PEARL/src/cpp/pearl.h>
#include "pro_pearl.h"

namespace py = pybind11;

PYBIND11_MODULE(pro_pearl, m) {
    m.doc() = "PRO_PEARL's implementation in C++";

    py::class_<adaptive_random_forest>(m, "adaptive_random_forest")
        .def(py::init<int,
                      int,
                      double,
                      double>())
        .def("init_data_source", &adaptive_random_forest::init_data_source)
        .def("get_next_instance", &adaptive_random_forest::get_next_instance)
        .def("get_cur_instance_label", &adaptive_random_forest::get_cur_instance_label)
        .def("delete_cur_instance", &adaptive_random_forest::delete_cur_instance)
        .def("predict", &adaptive_random_forest::predict)
        .def("train", &adaptive_random_forest::train);

    py::class_<pearl, adaptive_random_forest>(m, "pearl")
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
                      double,
                      bool,
                      bool>())
        .def("get_candidate_tree_group_size", &pearl::get_candidate_tree_group_size)
        .def("get_tree_pool_size", &pearl::get_tree_pool_size)
        .def("is_state_graph_stable", &pearl::is_state_graph_stable)
        .def("__repr__",
            [](const pearl &p) {
                return "<pearl.pearl has "
                    + std::to_string(p.get_tree_pool_size()) + " trees>";
            }
         );

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
                      double,
                      double>())
        .def_property_readonly("drift_detected", &pro_pearl::get_drift_detected)
        .def("find_last_actual_drift_point", &pro_pearl::find_last_actual_drift_point)
        .def("select_candidate_trees_proactively", &pro_pearl::select_candidate_trees_proactively)
        .def("adapt_state_proactively", &pro_pearl::adapt_state_proactively)
        .def("predict", &pro_pearl::predict)
        .def("adapt_state", &pro_pearl::adapt_state);
}
