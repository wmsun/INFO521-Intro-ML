import predictive_variance
import pytest
import numpy
import os


DATA_ROOT = 'data'
PATH_TO_SYNTH_DATA_SETS = os.path.join(DATA_ROOT, 'synth_data_sets.csv')


FIGURES_ROOT = 'figures'
PATH_TO_EX6_FN_NAME_BASE = os.path.join(FIGURES_ROOT, 'ex6_sample_fn_order')


@pytest.fixture
def exercise_results():
    return predictive_variance.exercise_6(orders=(1, 3, 5, 9),
                                          noise_var=6,
                                          xmin=-4.,
                                          xmax=5.,
                                          data_sets_source_path=PATH_TO_SYNTH_DATA_SETS,
                                          sampled_fn_figure_path_base=PATH_TO_EX6_FN_NAME_BASE)


SOLUTION = \
    ((numpy.array([2.39129649, 0.25974558]), 1.3523141756454908),
     (numpy.array([0.04104553, 5.26999519, 0.99164416, -0.50519182]), 27.15964772494143),
     (numpy.array([-3.63488170e-02, 5.22292432e+00, 1.02787500e+00, -4.94287758e-01,
                   -2.11982813e-03, -2.63898160e-04]), 26.87992606588288),
     (numpy.array([1.63903895e+00, 4.47567248e+00, -2.96005300e-01, 3.11369824e-02,
                   1.70803428e-01, -9.95084906e-02, 6.42880155e-03, 4.19586939e-03,
                   -1.04189344e-03, 7.67855380e-05]), -8.193279870175502))


def test_ex6_sampled_fns(exercise_results):
    ex6_results = exercise_results
    for (w_ex6, last_t_ex6), (w_sol, last_t_sol) in zip(ex6_results, SOLUTION):
        if w_ex6 is None:
            assert False
        else:
            assert numpy.allclose(w_ex6, w_sol)
        assert last_t_ex6 == pytest.approx(last_t_sol)
