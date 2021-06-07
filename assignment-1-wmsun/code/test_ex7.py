from code import hw1
import pytest


@pytest.fixture
def exercise_results():
    return hw1.exercise_7()


def test_seed(exercise_results):
    experiment_outcomes_1, experiment_outcomes_2, experiment_outcomes_3 = exercise_results
    assert experiment_outcomes_1 != experiment_outcomes_2
    assert experiment_outcomes_2 != experiment_outcomes_3
    assert experiment_outcomes_1 == experiment_outcomes_3


def test_expectted_sequence(exercise_results):
    experiment_outcomes_1, experiment_outcomes_2, experiment_outcomes_3 = exercise_results
    target1 = [0.167, 0.1655, 0.1696, 0.1697, 0.1547,
               0.163, 0.1686, 0.1676, 0.1557, 0.1646]
    target2 = [0.1714, 0.1666, 0.1625, 0.1689, 0.1691,
               0.164, 0.1699, 0.1656, 0.1654, 0.164]
    assert experiment_outcomes_1 == target1
    assert experiment_outcomes_2 == target2
    assert experiment_outcomes_3 == target1


def test_docstring():
    """
    You must always document your code!
    You must provide a descriptive docstring for
    the exercise_6 function.
    """
    assert hw1.exercise_7.__doc__ is not None
