import laplace_approx
import os
import pytest


FIGURES_ROOT = 'figures'
PATH_TO_LAPLACE1_FIG = os.path.join(FIGURES_ROOT, 'laplace_case1.png')
PATH_TO_LAPLACE2_FIG = os.path.join(FIGURES_ROOT, 'laplace_case2.png')
PATH_TO_LAPLACE3_FIG = os.path.join(FIGURES_ROOT, 'laplace_case3.png')


@pytest.fixture
def case1():
    return laplace_approx.plot_laplace_approx(alpha=5, beta=5, N=20, y=10, figure_path=PATH_TO_LAPLACE1_FIG)


def test_laplace_approx_case1(case1):
    gamma, delta, mu, sigma = case1
    assert gamma == pytest.approx(15)
    assert delta == pytest.approx(15)
    assert mu == pytest.approx(0.5)
    assert sigma == pytest.approx(0.0944911182523068)


@pytest.fixture
def case2():
    return laplace_approx.plot_laplace_approx(alpha=3, beta=15, N=10, y=3, figure_path=PATH_TO_LAPLACE2_FIG)
# return laplace_approx.plot_laplace_approx(alpha=3., beta=15., N=10., y=3.)


def test_laplace_approx_case2(case2):
    gamma, delta, mu, sigma = case2
    assert gamma == pytest.approx(6)
    assert delta == pytest.approx(22)
    assert mu == pytest.approx(0.19230769230769232)
    assert sigma == pytest.approx(0.07729201466043273)


@pytest.fixture
def case3():
    return laplace_approx.plot_laplace_approx(alpha=1, beta=30, N=10, y=3, figure_path=PATH_TO_LAPLACE3_FIG)
# return laplace_approx.plot_laplace_approx(alpha=1., beta=30., N=10., y=3.)


def test_laplace_approx_case3(case3):
    gamma, delta, mu, sigma = case3
    assert gamma == pytest.approx(4)
    assert delta == pytest.approx(37)
    assert mu == pytest.approx(0.07692307692307693)
    assert sigma == pytest.approx(0.042669245863479165)
