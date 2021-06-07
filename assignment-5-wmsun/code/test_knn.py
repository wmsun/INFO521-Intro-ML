import knn
import pytest
import os
import numpy


DATA_ROOT = 'data'
PATH_TO_KNN_BINARY_DATA = os.path.join(DATA_ROOT, 'knn_binary_data.csv')
PATH_TO_KNN_THREE_CLASS_DATA = os.path.join(DATA_ROOT, 'knn_three_class_data.csv')

PATH_TO_GT_2C_1 = os.path.join(DATA_ROOT, 'gt_2c_1.csv')
PATH_TO_GT_2C_5 = os.path.join(DATA_ROOT, 'gt_2c_5.csv')
PATH_TO_GT_2C_10 = os.path.join(DATA_ROOT, 'gt_2c_10.csv')
PATH_TO_GT_2C_59 = os.path.join(DATA_ROOT, 'gt_2c_59.csv')

PATH_TO_GT_3C_1 = os.path.join(DATA_ROOT, 'gt_3c_1.csv')
PATH_TO_GT_3C_5 = os.path.join(DATA_ROOT, 'gt_3c_5.csv')
PATH_TO_GT_3C_10 = os.path.join(DATA_ROOT, 'gt_3c_10.csv')
PATH_TO_GT_3C_59 = os.path.join(DATA_ROOT, 'gt_3c_59.csv')

FIGURES_ROOT = 'figures'


# -----------------------------------------------------------------------------

def compare_values(classes, ground_truth_path):
    gt_classes = numpy.loadtxt(ground_truth_path, delimiter=',')
    # numpy.savetxt(ground_truth_path[:-4] + '_predicted.csv', X=classes, fmt='%d')
    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
            assert classes[i][j] == gt_classes[i][j], \
                f'Mistmatch in classes prediction at i={i} j={j}: '\
                f'your knn predicted {classes[i][j]} '\
                f'but we expected {gt_classes[i][j]}'


# -----------------------------------------------------------------------------

@pytest.fixture
def exercise_results_knn_2c_1():
    x, t = knn.read_data(path=PATH_TO_KNN_BINARY_DATA)
    return knn.plot_decision_boundary(k=1, x=x, t=t, figures_root=FIGURES_ROOT, data_name='2c')


def test_knn2c_1(exercise_results_knn_2c_1):
    classes = exercise_results_knn_2c_1
    compare_values(classes, PATH_TO_GT_2C_1)


@pytest.fixture
def exercise_results_knn_2c_5():
    x, t = knn.read_data(path=PATH_TO_KNN_BINARY_DATA)
    return knn.plot_decision_boundary(k=5, x=x, t=t, figures_root=FIGURES_ROOT, data_name='2c')


def test_knn2c_5(exercise_results_knn_2c_5):
    classes = exercise_results_knn_2c_5
    compare_values(classes, PATH_TO_GT_2C_5)


@pytest.fixture
def exercise_results_knn_2c_10():
    x, t = knn.read_data(path=PATH_TO_KNN_BINARY_DATA)
    return knn.plot_decision_boundary(k=10, x=x, t=t, figures_root=FIGURES_ROOT, data_name='2c')


def test_knn2c_10(exercise_results_knn_2c_10):
    classes = exercise_results_knn_2c_10
    compare_values(classes, PATH_TO_GT_2C_10)


@pytest.fixture
def exercise_results_knn_2c_59():
    x, t = knn.read_data(path=PATH_TO_KNN_BINARY_DATA)
    return knn.plot_decision_boundary(k=59, x=x, t=t, figures_root=FIGURES_ROOT, data_name='2c')


def test_knn2c_59(exercise_results_knn_2c_59):
    classes = exercise_results_knn_2c_59
    compare_values(classes, PATH_TO_GT_2C_59)

# -----------------------------------------------------------------------------

@pytest.fixture
def exercise_results_knn_3c_1():
    x, t = knn.read_data(path=PATH_TO_KNN_THREE_CLASS_DATA)
    return knn.plot_decision_boundary(k=1, x=x, t=t, figures_root=FIGURES_ROOT, data_name='3c')


def test_knn3c_1(exercise_results_knn_3c_1):
    classes = exercise_results_knn_3c_1
    compare_values(classes, PATH_TO_GT_3C_1)


@pytest.fixture
def exercise_results_knn_3c_5():
    x, t = knn.read_data(path=PATH_TO_KNN_THREE_CLASS_DATA)
    return knn.plot_decision_boundary(k=5, x=x, t=t, figures_root=FIGURES_ROOT, data_name='3c')


def test_knn3c_5(exercise_results_knn_3c_5):
    classes = exercise_results_knn_3c_5
    compare_values(classes, PATH_TO_GT_3C_5)


@pytest.fixture
def exercise_results_knn_3c_10():
    x, t = knn.read_data(path=PATH_TO_KNN_THREE_CLASS_DATA)
    return knn.plot_decision_boundary(k=10, x=x, t=t, figures_root=FIGURES_ROOT, data_name='3c')


def test_knn3c_10(exercise_results_knn_3c_10):
    classes = exercise_results_knn_3c_10
    compare_values(classes, PATH_TO_GT_3C_10)


@pytest.fixture
def exercise_results_knn_3c_59():
    x, t = knn.read_data(path=PATH_TO_KNN_THREE_CLASS_DATA)
    return knn.plot_decision_boundary(k=59, x=x, t=t, figures_root=FIGURES_ROOT, data_name='3c')


def test_knn3c_59(exercise_results_knn_3c_59):
    classes = exercise_results_knn_3c_59
    compare_values(classes, PATH_TO_GT_3C_59)
