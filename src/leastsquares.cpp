#include <algorithm>
#include <iostream>
#include "leastsquares.h"
#include <pybind11/pybind11.h>
#include <Eigen/SVD>

using namespace std;
namespace py=pybind11;

LeastSquaresClassifier::LeastSquaresClassifier() {
}

void LeastSquaresClassifier::fit(Vector x, Vector y) {
    Matrix A = buildMatrix(x);
    this->coef = (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * y);
}

// TODO: podria usar ComputeThinU nomas?
Vector LeastSquaresClassifier::calculate(Matrix A, Vector b) {
    Eigen::BDCSVD<Matrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector c = svd.matrixU().transpose() * b;
    Vector z = Vector::Zero(A.cols());
    auto singularValues = svd.singularValues();
    for (int i = 0; i < svd.nonzeroSingularValues(); i++) {
        z(i) = c(i) / singularValues(i);
    }
    return svd.matrixV() * z;
}
