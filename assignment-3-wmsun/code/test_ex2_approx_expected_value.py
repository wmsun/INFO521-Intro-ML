import approx_expected_value
import pytest
import os


DATA_ROOT = 'data'
PATH_TO_RANDOM_UNIFORM_10000 = os.path.join(DATA_ROOT, 'rand_uniform_10000.txt')

FIGURES_ROOT = 'figures'
PATH_TO_FN_APPROX_FIG = os.path.join(FIGURES_ROOT, 'ex2_fn_approx.png')


@pytest.fixture
def exercise_results():
    return approx_expected_value.exercise_2(PATH_TO_RANDOM_UNIFORM_10000,
                                            PATH_TO_FN_APPROX_FIG)


def test_exercise_2(exercise_results):
    expected, estimate = exercise_results
    assert expected == pytest.approx(29.16)
    assert estimate == pytest.approx(29.19337786903892)
