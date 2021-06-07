import pi_sample_estimate
import math
import pytest


@pytest.fixture
def exercise_results():
    return pi_sample_estimate.estimate_pi(1000000)


def test_calculate_poisson_pmf_a(exercise_results):
    p = exercise_results
    tolerance = 0.005
    assert (math.pi - tolerance) < p < (math.pi + tolerance)
