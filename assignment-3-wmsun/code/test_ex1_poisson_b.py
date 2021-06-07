import poisson
import pytest


@pytest.fixture
def exercise_results():
    return poisson.calculate_poisson_pmf_b()


def test_calculate_poisson_pmf_a(exercise_results):
    p = exercise_results
    assert p == pytest.approx(0.2580236935530884)
