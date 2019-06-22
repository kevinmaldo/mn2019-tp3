#include <algorithm>
#include <iostream>
#include "leastsquares.h"
#include <pybind11/pybind11.h>
#include <Eigen/SVD>

using namespace std;
namespace py=pybind11;

LeastSquaresClassifier::LeastSquaresClassifier() {
}

void printMatrix(Matrix m) {
    stringstream ss;
    ss << m;
    py::print(ss.str());
}

Matrix LeastSquaresClassifier::buildMatrix(Vector x) {
    //Matrix A(x.rows(), 3);
    //A << x.cwiseProduct(x), x, Vector::Ones(x.rows());
    //return A;
    Matrix A(x.rows(), 2);
    A << x, Vector::Ones(x.rows());
    return A;
}

void LeastSquaresClassifier::fit(Vector x, Vector y) {
    Matrix A = buildMatrix(x);
    // printMatrix(A);
    this->coef = (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * y);
}

// TODO: podria usar ComputeThinU nomas?
Vector LeastSquaresClassifier::calculate(Matrix A, Vector y) {
    py::print(A.rows());
    py::print(A.cols());
    py::print(y.rows());
    Eigen::BDCSVD<Matrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector c = svd.matrixU().transpose() * y;
    Vector z = Vector::Zero(A.cols());
    auto singularValues = svd.singularValues();
    for (int i = 0; i < svd.nonzeroSingularValues(); i++) {
        z(i) = c(i) / singularValues(i);
    }
    return svd.matrixV() * z;
}

Vector LeastSquaresClassifier::predict(Vector x) {
    return buildMatrix(x) * this->coef;
}
