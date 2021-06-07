from code import cv
import pytest


@pytest.fixture
def demo_results():
    return cv.run_demo(seed=29)


def test_run_demo(demo_results):
    print(f'>>>>>> {demo_results}')
    best_poly, min_mean_log_cv_loss = demo_results
    assert best_poly == 3
    assert min_mean_log_cv_loss == pytest.approx(8.048721532610884)
