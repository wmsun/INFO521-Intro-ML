from code import hw1
import pytest
import numpy


@pytest.fixture
def exercise_results():
    return hw1.exercise_8()


def test_x(exercise_results):
    x, _, _, _, _, _, _, _, _, _, _ = exercise_results
    target = numpy.array([[0.07630829],
                          [0.77991879],
                          [0.43840923]])
    assert numpy.allclose(x, target)


def test_y(exercise_results):
    _, y, _, _, _, _, _, _, _, _, _ = exercise_results
    target = numpy.array([[0.72346518],
                          [0.97798951],
                          [0.53849587]])
    assert numpy.allclose(y, target)


def test_v1(exercise_results):
    _, _, v1, _, _, _, _, _, _, _, _ = exercise_results
    target = numpy.array([[0.79977347],
                          [1.7579083],
                          [0.9769051]])
    assert numpy.allclose(v1, target)


def test_v2(exercise_results):
    _, _, _, v2, _, _, _, _, _, _, _ = exercise_results
    target = numpy.array([[0.05520639],
                          [0.7627524],
                          [0.23608156]])
    assert numpy.allclose(v2, target)


def test_xT(exercise_results):
    _, _, _, _, xT, _, _, _, _, _, _ = exercise_results
    target = numpy.array([[0.07630829, 0.77991879, 0.43840923]])
    assert numpy.allclose(xT, target)


def test_v3(exercise_results):
    _, _, _, _, _, v3, _, _, _, _, _ = exercise_results
    target = numpy.array([[1.05404035]])
    assert numpy.allclose(v3, target)


def test_A(exercise_results):
    _, _, _, _, _, _, A, _, _, _, _ = exercise_results
    target = numpy.array([[0.50112046, 0.07205113, 0.26843898],
                          [0.4998825,  0.67923,    0.80373904],
                          [0.38094113, 0.06593635, 0.2881456]])
    assert numpy.allclose(A, target)


def test_v4(exercise_results):
    _, _, _, _, _, _, _, v4, _, _, _ = exercise_results
    target = numpy.array([[0.59511551, 0.56414944, 0.77366099]])
    assert numpy.allclose(v4, target)


def test_v5(exercise_results):
    _, _, _, _, _, _, _, _, v5, _, _ = exercise_results
    target = numpy.array([[1.39889083]])
    assert numpy.allclose(v5, target)


def test_v6(exercise_results):
    _, _, _, _, _, _, _, _, _, v6, _ = exercise_results
    target = numpy.array([[  6.31703988,  -0.1354985,   -5.50705724],
                          [  7.17645259,   1.86500328, -11.8877941 ],
                          [ -9.99359145,  -0.24763367,  13.47132266]])
    assert numpy.allclose(v6, target)


def test_v7(exercise_results):
    _, _, _, _, _, _, _, _, _, _, v7 = exercise_results
    target = numpy.array([[ 1.00000000e+00,  8.62138497e-18,  9.13821573e-17],
                          [ 1.00685164e-15,  1.00000000e+00, -5.72661470e-17],
                          [ 1.67793712e-16,  1.46395648e-17,  1.00000000e+00]])
    assert numpy.allclose(v7, target)


def test_docstring():
    """
    You must always document your code!
    You must provide a descriptive docstring for
    the exercise_6 function.
    """
    assert hw1.exercise_8.__doc__ is not None
