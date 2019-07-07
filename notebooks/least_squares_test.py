#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import airline

def test_linear():
    x = np.arange(0, 10, 0.1)
    delta = np.random.uniform(-5, 5, size=(x.shape[0],))
    y_true = 4*x + 3 + delta
    A = np.stack([x**0, x**1], axis=1)
    lstsq = airline.LeastSquaresClassifier()
    coeffs = lstsq.calculate(A, y_true)
    numpy_coeffs = np.linalg.lstsq(A, y_true, rcond=None)[0]
    print(np.allclose(coeffs, numpy_coeffs))

def test_cuadratic():
    x = np.arange(0, 10, 0.1)
    delta = np.random.uniform(-5, 5, size=(x.shape[0],))
    y_true = -2 * x**2 + 4*x + 3 + delta
    A = np.stack([x**0, x**1, x**2], axis=1)
    lstsq = airline.LeastSquaresClassifier()
    coeffs = lstsq.calculate(A, y_true)
    numpy_coeffs = np.linalg.lstsq(A, y_true, rcond=None)[0]
    print(np.allclose(coeffs, numpy_coeffs))

def test_sinusoidal():
    x = np.arange(0, 10, 0.1)
    delta = np.random.uniform(-1, 1, size=(x.shape[0],))
    y_true = 2 * np.sin(x + 2) + x + delta
    A = np.stack([x**0, x**1, np.sin(x), np.cos(x)], axis=1)
    lstsq = airline.LeastSquaresClassifier()
    coeffs = lstsq.calculate(A, y_true)
    numpy_coeffs = np.linalg.lstsq(A, y_true, rcond=None)[0]
    print(np.allclose(coeffs, numpy_coeffs))

if __name__ == "__main__":
    test_linear()
    test_cuadratic()
    test_sinusoidal()
