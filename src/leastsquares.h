#pragma once

#include "types.h"


class LeastSquaresClassifier {
public:
    LeastSquaresClassifier();

    void fit(Vector x, Vector y);

    Vector calculate(Matrix A, Vector y);

    Vector predict(Vector x);

private:

    Vector coef;

    Matrix buildMatrix(Vector x);

};
