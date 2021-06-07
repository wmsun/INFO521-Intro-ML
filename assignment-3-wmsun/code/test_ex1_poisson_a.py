import poisson
import pytest


@pytest.fixture
def exercise_results():
    return poisson.calculate_poisson_pmf_a()


def test_calculate_poisson_pmf_a(exercise_results):
    p = exercise_results
    assert p == pytest.approx(0.7419763064469116)
