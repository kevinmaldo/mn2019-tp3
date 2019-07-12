#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "leastsquares.h"

namespace py=pybind11;

PYBIND11_MODULE(airline, m) {
    py::class_<LeastSquaresClassifier>(m, "LeastSquaresClassifier")
        .def(py::init<>())
        .def("fit", &LeastSquaresClassifier::fit)
        .def("predict", &LeastSquaresClassifier::predict)
        .def("calculate", &LeastSquaresClassifier::calculate);
}
