import numpy
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter


# --------------------------------------------------------------------------
# Control exercise execution
# --------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_EXERCISE = True

# --------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------

DATA_ROOT = os.path.join('..', 'data')
PATH_TO_KNN_BINARY_DATA = os.path.join(DATA_ROOT, 'knn_binary_data.csv')
PATH_TO_KNN_THREE_CLASS_DATA = os.path.join(DATA_ROOT, 'knn_three_class_data.csv')

FIGURES_ROOT = os.path.join('..', 'figures')

# Create color map
CMAP_LIGHT = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])


# --------------------------------------------------------------------------
# KNN Classifier
# --------------------------------------------------------------------------

def read_data(path, d=','):
    """
    Read 2-dimensional real-valued features with associated class labels
    :param path: path to csv file
    :param d: delimiter
    :return: x=array of features, t=class labels
    """
    arr = numpy.genfromtxt(path, delimiter=d, dtype=None)
    length = len(arr)
    x = numpy.zeros(shape=(length, 2))
    t = numpy.zeros(length, dtype=int)
    for i, (x1, x2, tv) in enumerate(arr):
        x[i, 0] = x1
        x[i, 1] = x2
        t[i] = int(tv)
    return x, t


def plot_data(x, t, new_figure=True, save_path=None):
    """
    Plot the data as a scatter plot
    :param x:
    :param t:
    :param new_figure: Flag for whether to create a new figure; don't when plotting on top of existing fig.
    :param save_path:
    :return:
    """
    # Plot the binary data
    ma = ['o', 's', 'v']
    fc = ['r', 'g', 'b']  # np.array([0, 0, 0]), np.array([1, 1, 1])]
    tv = numpy.unique(t.flatten())  # an array of the unique class labels
    if new_figure:
        plt.figure()
    for i in range(tv.shape[0]):
        pos = (t == tv[i]).nonzero()  # returns a boolean vector mask for selecting just the instances of class tv[i]
        plt.scatter(numpy.asarray(x[pos, 0]), numpy.asarray(x[pos, 1]), marker=ma[i], facecolor=fc[i])

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    if save_path is not None:
        plt.savefig(save_path, fmt='png')


# --------------------------------------------------------------------------
# KNN core

def knn(p, k, x, t):
    """
    K-Nearest Neighbors classifier.  Return the most frequent class among the k nearest points
    :param p: point to classify (assumes 2-dimensional)
    :param k: number of nearest neighbors
    :param x: array of observed 2-dimensional points
    :param t: array of target labels (corresponding to points)
    :return: the top class label
    """

    # Number of instances in data set
    N = x.shape[0]

    Euclidean_Distance = numpy.square(x - p) #Euclidean distance
    dis = numpy.sum(Euclidean_Distance, axis=1) #sum of the euclidean distance
    inds = numpy.argsort(dis)[:k] #sort the indices of the distance array
    tgt_cat = Counter([t[i] for i in inds]) #count the times of equivalent target labels
    top_class = max(tgt_cat, key= tgt_cat.get) #top class among the k nearest points


    #top_class = 0

    return top_class


# --------------------------------------------------------------------------
# Plot the decision boundary

def plot_decision_boundary(k, x, t, granularity=100, figures_root='../figures', data_name=None):
    """
    Given data (observed x and labels t) and choice k of nearest neighbors,
    plots the decision boundary based on a grid of classifications over the
    feature space.
    :param k: number of nearest neighbors
    :param x: array of observed 2-dimensional points
    :param t: array of target labels (corresponding to points)
    :param granularity: controls granularity of the meshgrid
    :param figures_root: path to figures root
    :param data_name: optional name for dataset, used in figure generation
    :return:
    """
    print(f'KNN for K={k}')

    # Initialize meshgrid to be used to store the class prediction values
    # this is used for computing and plotting the decision boundary contour

    pointsX = numpy.linspace(numpy.min(x[:, 0]) - 0.1, numpy.max(x[:, 0]) + 0.1, granularity)
    pointsY = numpy.linspace(numpy.min(x[:, 1]) - 0.1, numpy.max(x[:, 1]) + 0.1, granularity)

    Xv, Yv = numpy.meshgrid(pointsX, pointsY)

    # Calculate KNN classification for every point in meshgrid
    classes = numpy.zeros(shape=(Xv.shape[0], Xv.shape[1]))
    for i in range(Xv.shape[0]):
        for j in range(Xv.shape[1]):
            c = knn(numpy.array([Xv[i][j], Yv[i][j]]), k, x, t)
            # print('{0} {1} {2}'.format(i, j, c))
            classes[i][j] = c

    # plot the binary decision boundary contour
    plt.figure()
    plt.pcolormesh(Xv, Yv, classes, cmap=CMAP_LIGHT)
    ti = f'KNN with K = {k}'
    plt.title(ti)
    plt.draw()

    save_path = None
    if data_name is not None:
        save_path = os.path.join(figures_root, f'knn_{data_name}_k={k}')
    # else:
    #     save_path = os.path.join(figures_root, f'knn_k={k}')

    # plot the data (on top of the decision boundary color mesh)
    plot_data(x, t, new_figure=False, save_path=save_path)

    return classes


def run_knn_script(data_path, figures_root, data_name):
    x, t = read_data(data_path)

    # Loop over different neighborhood values K
    for k in [1, 5, 10, 59]:
        plot_decision_boundary(k, x, t, figures_root=figures_root, data_name=data_name)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if __name__ == "__main__":
    if RUN_EXERCISE:
        run_knn_script(data_path=PATH_TO_KNN_BINARY_DATA, figures_root=FIGURES_ROOT, data_name='2c')
        run_knn_script(data_path=PATH_TO_KNN_THREE_CLASS_DATA, figures_root=FIGURES_ROOT, data_name='3c')
        plt.show()


