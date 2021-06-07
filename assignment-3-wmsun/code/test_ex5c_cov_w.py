import predictive_variance
import pytest
import numpy
import os


X = numpy.array([[1., -2.30444123],
                 [1., 4.92258244],
                 [1., 2.06230454],
                 [1., 3.457873],
                 [1., 1.73525716],
                 [1., 1.75596156],
                 [1., 4.62011482],
                 [1., 1.91417156],
                 [1., 2.50759722],
                 [1., -3.028605],
                 [1., 2.94043152],
                 [1., -1.04502716],
                 [1., -3.03197986],
                 [1., -3.33886741],
                 [1., 4.22204963],
                 [1., -2.32779827],
                 [1., -3.33044065],
                 [1., 2.19358405],
                 [1., 4.51864485],
                 [1., 2.88489857],
                 [1., 4.15356212],
                 [1., 1.67978355],
                 [1., 3.93199799],
                 [1., -2.61346468],
                 [1., 4.76432263],
                 [1., 2.16134283],
                 [1., -3.02595779],
                 [1., -3.26310816],
                 [1., 2.56888303],
                 [1., 4.28624582],
                 [1., 1.48506739],
                 [1., 4.32580881],
                 [1., 1.26211007],
                 [1., 1.57361947]])

w = numpy.array([6.31398205, -0.72540592])

t = numpy.array([2.53788261, -9.03876506, 6.3046432, 9.93977522, 2.53244664, 6.88888787,
                 -8.66109976, 11.85530387, 7.55075759, 10.91767088, 7.93294935, -2.97935296,
                 1.33979277, 14.89583933, 4.88225172, 0.83771979, 11.47070662, 9.61560325,
                 -3.60524702, 11.86294607, 2.36891125, 9.18121807, 7.55686174, 3.81824221,
                 -7.45711442, 14.48827314, 5.47652938, 9.91784932, 12.7403704, -1.13251662,
                 9.69199017, 2.8873025, 9.62115036, 6.06906844])


@pytest.fixture
def exercise_results_ex5c_predictive_variance():
    return predictive_variance.calculate_cov_w(X, w, t)


def test_ex5c_predictive_variance(exercise_results_ex5c_predictive_variance):
    ex5c_cov_w = exercise_results_ex5c_predictive_variance
    ex5c_covw_solution = numpy.array([[ 1.34170567, -0.18152876], [-0.18152876,  0.1383277 ]])
    assert ex5c_cov_w == pytest.approx(ex5c_covw_solution)


DATA_ROOT = 'data'
PATH_TO_SYNTH_DATA = os.path.join(DATA_ROOT, 'synth_data.csv')

FIGURES_ROOT = 'figures'
PATH_TO_EX5C_FN_NAME_BASE = os.path.join(FIGURES_ROOT, 'ex5c_sample_fn_order')

xmin = -4.
xmax = 5.
x, t = predictive_variance.read_data_from_file(PATH_TO_SYNTH_DATA)
xmin_remove = -1
xmax_remove = 1
pos = ((x >= xmin_remove) & (x <= xmax_remove)).nonzero()
x = numpy.delete(x, pos, 0)
t = numpy.delete(t, pos, 0)
orders = (1, 3, 5, 9)
num_function_samples = 20


# The following test is here just to generate your figures when the tests are run
# This is expected to pass right a way

@pytest.fixture
def exercise_results_ex5c():
    return predictive_variance.plot_functions_sampling_from_covw\
        (x, t, orders, num_function_samples,
         xmin, xmax, PATH_TO_EX5C_FN_NAME_BASE)


def test_ex5c_plot_functions_sampling_from_covw(exercise_results_ex5c):
    _ = exercise_results_ex5c
    assert os.path.exists('figures/ex5c_sample_fn_order-1.png')
    assert os.path.exists('figures/ex5c_sample_fn_order-3.png')
    assert os.path.exists('figures/ex5c_sample_fn_order-5.png')
    assert os.path.exists('figures/ex5c_sample_fn_order-9.png')
