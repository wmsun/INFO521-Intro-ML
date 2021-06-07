from code import hw1
import pytest
import numpy


@pytest.fixture
def exercise_results():
    return hw1.exercise_9('data/X.txt', 'data/w.txt')


def test_read_data(exercise_results):
    X, w, _, _, _, _, _ = exercise_results
    target_X = numpy.array([[27., 13.],
                            [8.,  12.],
                            [9.,  3. ],
                            [14., 15.],
                            [18., 4. ]])
    target_w = numpy.array([3., 7.])
    assert numpy.allclose(X, target_X)
    assert numpy.allclose(w, target_w)


def test_x_column_vectors(exercise_results):
    _, _, x_n1, x_n2, _, _, _ = exercise_results
    target_x_n1 = numpy.array([27.,  8.,  9., 14., 18.])
    target_x_n2 = numpy.array([13., 12.,  3., 15.,  4.])
    assert numpy.allclose(x_n1, target_x_n1)
    assert numpy.allclose(x_n2, target_x_n2)


def test_scalar_result(exercise_results):
    _, _, _, _, scalar_result, _, _ = exercise_results
    assert scalar_result == pytest.approx(71885.0)


def test_XX(exercise_results):
    _, _, _, _, _, XX, _ = exercise_results
    targetXX = numpy.array([[1394., 756.],
                            [ 756., 563.]])
    assert numpy.allclose(XX, targetXX)


def test_wXXw(exercise_results):
    _, _, _, _, _, _, wXXw = exercise_results
    assert wXXw == pytest.approx(71885.0)


def test_docstring():
    """
    You must always document your code!
    You must provide a descriptive docstring for
    the exercise_6 function.
    """
    assert hw1.exercise_9.__doc__ is not None
