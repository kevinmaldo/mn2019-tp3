#include <algorithm>
#include <iostream>
#include "leastsquares.h"
#include <pybind11/pybind11.h>

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

Vector LeastSquaresClassifier::predict(Vector x) {
    return buildMatrix(x) * this->coef;
}
