from code import fitpoly
import numpy
import pytest
import os


@pytest.fixture
def exercise_results():
    return fitpoly.exercise_2(os.path.join('data', 'synthdata2019.csv'), os.path.join('figures', 'synthetic2019-3rd-poly.png'))


def test_ex2_w(exercise_results):
    target = numpy.array([-11.64034843, 6.54214409, 3.14150641, -2.27581331])
    w = exercise_results
    assert numpy.allclose(w, target)


def test_synthetic_poly_figure_exists():
    assert os.path.exists(os.path.join('figures', 'synthetic2019-3rd-poly.png'))
