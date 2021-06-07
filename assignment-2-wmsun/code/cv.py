# ISTA 421 / INFO 521 Fall 2019, HW 2, Exercises 1 and 2
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 2
# of the current year (2019).
# You are NOT permitted to share this file with other students outside of
# this course year. Doing so will be considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

# NOTE:
# When summarizing log errors, DO NOT take the mean of the log
# instead, first take the mean of the errors, then take the log of the mean

import os
import numpy
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_DEMO = False
RUN_EXERCISE_3_5FOLD = False
RUN_EXERCISE_3_LOOCV = True

# -----------------------------------------------------------------------------
# Global paths
# -----------------------------------------------------------------------------

DATA_ROOT = os.path.join('..', 'data')
DATA_SYNTH2019 = os.path.join(DATA_ROOT, 'synthdata2019.csv')

FIGURES_ROOT = os.path.join('..', 'figures')
PATH_TO_SYNTH_5FOLDCV_FIG = os.path.join(FIGURES_ROOT, 'synthetic2019-5fold-CV.png')
PATH_TO_SYNTH_LOOCV_FIG = os.path.join(FIGURES_ROOT, 'synthetic2019-LOOCV.png')


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

# The following permutation tools are used below in run_cv() as a convenience
# for randomly sorting the order of the data (more specifically, the indices
# of the input data)

def random_permutation_matrix(n):
    """
    Generate a permutation matrix: an NxN matrix in which each row
    and each column has only one 1, with 0's everywhere else.
    See: https://en.wikipedia.org/wiki/Permutation_matrix
    :param n: size of the square permutation matrix
    :return: NxN permutation matrix
    """
    rows = numpy.random.permutation(n)
    cols = numpy.random.permutation(n)
    m = numpy.zeros((n, n))
    for r, c in zip(rows, cols):
        m[r][c] = 1
    return m


def permute_rows(X, P=None):
    """
    Permute the rows of a 2-d array (matrix) according to
    permutation matrix P: left-multiply X by P
    If no P is provided, a random permutation matrix is generated.
    :param X: 2-d array
    :param P: Optional permutation matrix; default=None
    :return: new version of X with rows permuted according to P
    """
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return numpy.dot(P, X)


def permute_cols(X, P=None):
    """
    Permute the columns of a 2-d array (matrix) according to
    permutation matrix P: right-multiply X by P
    If no P is provided, a random permutation matrix is generated.
    :param X: 2-d array
    :param P: Optional permutation matrix; default=None
    :return: new version of X with columns permuted according to P
    """
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return numpy.dot(X, P)



def read_data(filepath, d=','):
    """
    Wrapper to genfromtext a numpy.array of the data.

    :param filepath: sting representing path to data file to load.
    :param d: char string representing delimiter separating values on
              a line in data file.
    """
    return numpy.genfromtxt(filepath, delimiter=d, dtype=None)


def plot_data(x, t, title='Data'):
    """
    Plot single input feature x data with corresponding response
    values t as a scatter plot.

    :param x: sequence of 1-dimensional input data values (numbers)
    :param t: sequence of 1-dimensional responses
    :param title: title of plot (default 'Data')
    :return: None
    """
    plt.figure()  # Create a new figure object for plotting
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(title)
    plt.pause(.1)  # required on some systems to allow rendering


def plot_model(x, w, color='r'):
    """
    Plot the function (curve) of an n-th order polynomial model
    with one input (x):
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n
    This works by creating a set of x-axis (plotx) points and
    then uses the model parameters w to determine the corresponding
    t-axis (plott) points on the model curve.

    :param x: sequence of 1-dimensional input data values (numbers)
    :param w: n-dimensional sequence of model parameters: w0, w1, w2, ..., wn
    :param color: specifies the color of the model curve
    :return: the plotx and plott values for the plotted curve
    """
    # NOTE: this assumes a figure() object has already been created.

    # plotx represents evenly-spaced set of 100 points on the x-axis.
    # This is used for creating a relatively "smooth" model curve plot.
    # Includes points a little before the min x input (-0.25)
    # and a little after the max x input (+0.25)
    plotx = numpy.linspace(min(x)-0.25, max(x)+0.25, 100)

    # plotX (note that python is case sensitive, so this is *not*
    # the same as plotx with a lower-case x) is the "design matrix"
    # for our model curve inputs represented in plotx.
    # We need to do the same computation as we do when doing
    # model fitting (as in fitpoly(), below), except that we
    # don't need to infer (by the normal equations) the values
    # of w, as they are given here as input.
    # plotx.shape[0] ensures we create a matrix with the number of
    # rows corresponding to the number of points in plotx (this will
    # still work even if we change the number of plotx points to
    # something other than 100)
    plotX = numpy.zeros((plotx.shape[0], w.size))

    # populate the design matrix plotX
    for k in range(w.size):
        plotX[:, k] = numpy.power(plotx, k)

    # Take the dot (inner) product of the design matrix plotX and the
    # parameter vector w
    plott = numpy.dot(plotX, w)

    # plot the x (plotx) and t (plott) values in red
    plt.plot(plotx, plott, color=color, linewidth=2)

    plt.pause(.1)  # required on some systems to allow rendering
    return plotx, plott


def scale01(x):
    """
    HELPER FUNCTION: only needed if you are working with relatively
    large x values.  This is NOT needed for problems 1, 2 and 4.

    Mathematically, the sizes of the values of variables (such as x)
    could be arbitrarily large.  However, when we go to manipulate
    those values on a computer, we need to be careful.  E.g., in
    these exercises we are taking powers of values and if the
    values are large, then taking a large power of the variable
    may exceed what can be represented numerically in the computer.

    For example, in the Olympics data (both men's and women's),
    the input x values are years in the 1000's.  If your model
    is, say, polynomial order 5, then you're taking a large
    number to the power of 5, which is the order of a quadrillion!
    Python floating point numbers have trouble representing this
    many significant digits.

    This function here scales the input data to be the range [0, 1]
    (i.e., between 0 and 1, inclusive).

    :param x: sequence of 1-dimensional input data values (numbers)
    :return: x values linearly scaled to range [0, 1]
    """
    x_min = min(x)
    x_range = max(x) - x_min
    return (x - x_min) / x_range


# -----------------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------------

def generate_synthetic_data(N, w, xmin=-5, xmax=5, sigma=150):
    """
    Sample N (x,t) points where x is uniform between xmin and xmax,
     and t is Gaussian (Normal) distributed with the mean provided by the
     generating model (polynomial in x with exponents specified by w)
     and standard deviation sigma.
    :param N: Number of sample points
    :param w: numpy array (1d) representing generating model parameters
    :param xmin: x minimum
    :param xmax: x maximum
    :param sigma: standard deviation
    :return: x array and t array, both of size N
    """
    # generate N random input x points between [xmin, xmax]
    x = (xmax - xmin) * numpy.random.rand(N) + xmin

    # generate response with Gaussian random noise
    X = numpy.zeros((x.size, w.size))
    for k in range(w.size):
        X[:, k] = numpy.power(x, k)
    # randn samples from a "Standard Normal" (0 mean, 1 stdev) distribution.
    t = numpy.dot(X, w) + sigma * numpy.random.randn(x.shape[0])

    return x, t


def plot_data_and_model(x, t, w,
                        title='Plot of synthetic data; green curve is original generating function',
                        filepath=None):
    plot_data(x, t)
    plt.title(title)
    plot_model(x, w, color='g')
    if filepath:
        plt.savefig(filepath, format='png')




# -----------------------------------------------------------------------------

def plot_cv_results(train_loss, cv_loss, ind_loss=None, log_scale_p=False,
                    plot_title='Mean Square Error Loss', filepath=None):
    """
    Helper function to plot the results of cross-validation.
    You can use this when you have all three types of loss, or
    if you just have train and cv loss (in this case, set ind_loss to None)
    :param train_loss:
    :param cv_loss:
    :param ind_loss: If None, only display train_loss and cv_loss
    :param log_scale_p: Whether the plots are in log-scale
    :param plot_title: Title for the plot
    :param filepath: If specified, save pdf file
    :return:
    """

    if ind_loss is not None:
        figw, figh, mydpi = 1200, 420, 96
        plt.figure(figsize=(figw / mydpi, figh / mydpi), dpi=mydpi)
    else:
        figw, figh, mydpi = 800, 420, 96
        plt.figure(figsize=(figw / mydpi, figh / mydpi), dpi=mydpi)

    # plt.figure()
    plt.suptitle(plot_title)

    if log_scale_p:
        ylabel = 'Log MSE Loss'
    else:
        ylabel = 'MSE Loss'

    x = numpy.arange(0, train_loss.shape[0])

    # put y-axis on same scale for all plots
    if ind_loss is not None:
        min_ylim = min(min(train_loss), min(cv_loss), min(ind_loss))
    else:
        min_ylim = min(min(train_loss), min(cv_loss))
    min_ylim = int(numpy.floor(min_ylim))
    if ind_loss is not None:
        max_ylim = max(max(train_loss), max(cv_loss), max(ind_loss))
    else:
        max_ylim = max(max(train_loss), max(cv_loss))
    max_ylim = int(numpy.ceil(max_ylim))

    # Plot training loss
    if ind_loss is not None:
        plt.subplot(131)
    else:
        plt.subplot(121)
    plt.plot(x, train_loss, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('CV Train Loss')
    plt.pause(.1) # required on some systems so that rendering can happen
    plt.ylim(min_ylim, max_ylim)

    # Plot CV loss
    if ind_loss is not None:
        plt.subplot(132)
    else:
        plt.subplot(122)
    plt.plot(x, cv_loss, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('CV Test Loss')
    plt.pause(.1) # required on some systems so that rendering can happen
    plt.ylim(min_ylim, max_ylim)

    # Plot independent-test loss
    if ind_loss is not None:
        plt.subplot(133)
        plt.plot(x, ind_loss, linewidth=2)
        plt.xlabel('Model Order')
        plt.ylabel(ylabel)
        plt.title('Independent Test Loss')
        plt.pause(.1) # required on some systems so that rendering can happen
        plt.ylim(min_ylim, max_ylim)

    if ind_loss is not None:
        plt.subplots_adjust(right=0.95, wspace=0.4)
    else:
        plt.subplots_adjust(right=0.95, wspace=0.25, bottom=0.2)

    plt.draw()

    if filepath:
        plt.savefig(filepath, format='pdf')


# -----------------------------------------------------------------------------
# The Cross-Validation DEMO (from FCML Ch 1, pp.31-32)
# -----------------------------------------------------------------------------

def run_cv(K, maxorder, x, t, independent_testx, independent_testt,
           randomize_data=False, title='CV'):
    """
    Implements the core of the cross-validation demonstration
    from FCML Ch 1, pp.31-32.
    :param K: Number of folds
    :param maxorder: Integer representing the highest polynomial order
    :param x: input data (1d observations)
    :param t: target data
    :param independent_testx: independent test input data
    :param independent_testt: independent test target data
    :param randomize_data: Boolean (default False) whether to randomize the order of the data
    :param title: Title for plots of results
    :return:
    """

    N = x.shape[0]  # number of data points

    # Use when you want to ensure the order of the data has been
    # randomized before splitting into folds
    # Note that in the simple demo here, we know that the data is
    # already in random order, so the run_demo script sets
    # randomize_data to False.
    # However, if you use this function more generally for new data,
    # you may need to ensure you're randomizing the order of the data!
    # Bottom line: when you are not 100% sure the data order has been
    # randomized, you should randomize the order!
    # Protip: always make sure that you use the *same* randomized
    # re-ordering of the input features x and response t -- that is
    # why the script below uses the same random permutation matrix P
    # to permute the rows of x and t. Otherwise you scramble the
    # relationship between your input features x and their associated
    # response t!
    if randomize_data:
        # use the same permutation P on both x and t, otherwise they'll
        # each be in different orders!
        P = random_permutation_matrix(x.shape[0])
        x = permute_rows(x, P)
        t = permute_rows(t, P)

    # Storage for the design matrix used during training
    # Here we create the design matrix to hold the maximum sized polynomial order
    # When computing smaller model polynomial orders below, we'll just use
    # the first 'k+1' columns for the k-th order polynomial.  This way we don't
    # have to keep creating new design matrix arrays.
    X = numpy.zeros((x.shape[0], maxorder + 1))

    # Design matrix for independent test data
    independent_testX = numpy.zeros((independent_testx.shape[0], maxorder + 1))

    # Create approximately equal-sized fold indices
    # These correspond to indices in the design matrix (X) rows
    # (where each row represents one training input x)
    fold_indices = list(map(lambda x: int(x), numpy.linspace(0, N, K + 1)))

    # storage for recording loss across model polynomial order
    # rows = fold loss (each row is the loss for one fold)
    # columns = model polynomial order
    cv_loss = numpy.zeros((K, maxorder + 1))     # cross-validation loss
    train_loss = numpy.zeros((K, maxorder + 1))  # training loss
    ind_loss = numpy.zeros((K, maxorder + 1))    # independent test loss

    # iterate over model polynomial orders
    for p in range(maxorder + 1):

        # Augment the input data by the polynomial model order
        # E.g., 2nd-order polynomial model takes input x to the 0th, 1st, and 2nd power
        X[:, p] = numpy.power(x, p)

        # ... do the same for the independent test data
        independent_testX[:, p] = numpy.power(independent_testx, p)

        # iterate over cross-validation folds
        for fold in range(K):
            # Partition the data
            # foldX, foldt contains the data for just one fold being held out
            # trainX, traint contains all other data

            foldX = X[fold_indices[fold]:fold_indices[fold+1], 0:p+1]
            foldt = t[fold_indices[fold]:fold_indices[fold+1]]

            # safely copy the training data (so that deleting doesn't remove the original
            trainX = numpy.copy(X[:, 0:p + 1])
            # remove the fold x from the training set
            trainX = numpy.delete(trainX, numpy.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

            # safely copy the training data (so that deleting doesn't remove the original
            traint = numpy.copy(t)
            # remove the fold t from the training set
            traint = numpy.delete(traint, numpy.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

            # find the least mean squares fit to the training data
            # NOTE: In this case, you do not want to directly use your fitpoly function
            #       from fitpoly.py here because that implementation constructs the design
            #       matrix X, but here you already have your design matrix X.
            ### YOUR CODE HERE
            w = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(trainX.T, trainX)), trainX.T), traint)  # Calculate w vector (as an numpy.array)

            # This conditional returns until you implement the calculation of w
            if w is None:
                return None, None

            # calculate and record the mean squared losses

            train_pred = numpy.dot(trainX, w)  # model predictions on training data
            train_loss[fold, p] = numpy.mean(numpy.power(train_pred - traint, 2))

            fold_pred = numpy.dot(foldX, w)  # model predictions on held-out fold
            cv_loss[fold, p] = numpy.mean(numpy.power(fold_pred - foldt, 2))

            ind_pred = numpy.dot(independent_testX[:, 0:p + 1], w)   # model predictions on independent test data
            ind_loss[fold, p] = numpy.mean(numpy.power(ind_pred - independent_testt, 2))

    # The loss values can get quite large, so take the log for display purposes

    # Ensure taking log of the mean (not mean of the log!)
    mean_train_loss = numpy.mean(train_loss, 0)
    mean_cv_loss = numpy.mean(cv_loss, 0)
    mean_ind_loss = numpy.mean(ind_loss, 0)

    log_mean_train_loss = numpy.log(mean_train_loss)
    log_mean_cv_loss = numpy.log(mean_cv_loss)
    log_mean_ind_loss = numpy.log(mean_ind_loss)

    print('\n----------------------\nResults for {0}'.format(title))
    print('log_mean_train_loss:\n{0}'.format(log_mean_train_loss))
    print('log_mean_cv_loss:\n{0}'.format(log_mean_cv_loss))
    print('log_mean_ind_loss:\n{0}'.format(log_mean_ind_loss))

    min_mean_log_cv_loss = min(log_mean_cv_loss)
    # TODO: has to be better way to get the min index...
    best_poly = [i for i, j in enumerate(log_mean_cv_loss) if j == min_mean_log_cv_loss][0]

    print('minimum mean_log_cv_loss of {0} for order {1}'.format(min_mean_log_cv_loss, best_poly))

    # Plot log scale loss results
    plot_cv_results(log_mean_train_loss, log_mean_cv_loss, log_mean_ind_loss,
                    log_scale_p=True, plot_title=title)

    # Uncomment to plot direct-scale mean loss results
    # plot_cv_results(mean_train_loss, mean_cv_loss, mean_ind_loss, log_scale_p=False)

    return best_poly, min_mean_log_cv_loss


# -----------------------------------------------------------------------------

def run_demo(seed=None):
    """
    Top-level script to run the cv demo from FCML Ch 1, pp.31-32.
    :param seed: (default None) When not None, integer for setting
        random number generator seed
    """

    # Set the random seed for reproducibility
    # Changing the seed (or not setting at all) will result in
    # different results from one run to the next.
    # It is expected that different results may vary quite a bit
    # from the
    if seed is not None:
        numpy.random.seed(seed=seed)

    # Parameters for synthetic data model (the model from which the
    # data is sampled)
    w = numpy.array([0, 1, 5, 2])  # t = x + 5x^2 + 2x^3 + N(0, sigma)
    xmin = -5
    xmax = 5
    sigma = 50

    # Generate synthetic data
    x, t = generate_synthetic_data(30, w, xmin=xmin, xmax=xmax, sigma=sigma)
    testx, testt, = generate_synthetic_data(1000, w, xmin=xmin, xmax=xmax, sigma=sigma)

    plot_data_and_model(x, t, w)

    K = 10

    best_poly, min_mean_log_cv_loss = \
        run_cv( K, 7, x, t, testx, testt, randomize_data=False, title=f'{K}-fold CV' )

    print(f'run_demo(): best_polynomial model order: {best_poly}')
    print(f'run_demo(): min_mean_log_cv_loss: {min_mean_log_cv_loss}')

    return best_poly, min_mean_log_cv_loss

def run_cv_solution(K, maxorder, x, t, randomize_data=True, title='CV'):
    """
    :param K: Number of folds
    :param maxorder: Integer representing the highest polynomial order
    :param x: input data (1d observations)
    :param t: target data
    :param randomize_data: Boolean (default False) whether to randomize the order of the data
    :param title: Title for plots of results
    :return:
    """

    N = x.shape[0]  # number of data points

    if randomize_data:
        # use the same permutation P on both x and t, otherwise they'll
        # each be in different orders!
        P = random_permutation_matrix(x.shape[0])
        x = permute_rows(x, P)
        t = permute_rows(t, P)

    # Storage for the design matrix used during training
    # Here we create the design matrix to hold the maximum sized polynomial order
    # When computing smaller model polynomial orders below, we'll just use
    # the first 'k+1' columns for the k-th order polynomial.  This way we don't
    # have to keep creating new design matrix arrays.
    X = numpy.zeros((x.shape[0], maxorder + 1))

    # Create approximately equal-sized fold indices
    # These correspond to indices in the design matrix (X) rows
    # (where each row represents one training input x)
    fold_indices = list(map(lambda x: int(x), numpy.linspace(0, N, K + 1)))

    # storage for recording loss across model polynomial order
    # rows = fold loss (each row is the loss for one fold)
    # columns = model polynomial order
    cv_loss = numpy.zeros((K, maxorder + 1))  # cross-validation loss
    train_loss = numpy.zeros((K, maxorder + 1))  # training loss
    #bestX = numpy.zeros((K, maxorder + 1)) #store values of w
    # iterate over model polynomial orders
    for p in range(maxorder + 1):

        # Augment the input data by the polynomial model order
        # E.g., 2nd-order polynomial model takes input x to the 0th, 1st, and 2nd power
        X[:, p] = numpy.power(x, p)

        # iterate over cross-validation folds
        for fold in range(K):
            # Partition the data
            # foldX, foldt contains the data for just one fold being held out
            # trainX, traint contains all other data

            foldX = X[fold_indices[fold]:fold_indices[fold + 1], 0:p + 1]
            foldt = t[fold_indices[fold]:fold_indices[fold + 1]]

            # safely copy the training data (so that deleting doesn't remove the original
            trainX = numpy.copy(X[:, 0:p + 1])
            # remove the fold x from the training set
            trainX = numpy.delete(trainX, numpy.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

            # safely copy the training data (so that deleting doesn't remove the original
            traint = numpy.copy(t)
            # remove the fold t from the training set
            traint = numpy.delete(traint, numpy.arange(fold_indices[fold], fold_indices[fold + 1]), 0)

            # find the least mean squares fit to the training data
            # NOTE: In this case, you do not want to directly use your fitpoly function
            #       from fitpoly.py here because that implementation constructs the design
            #       matrix X, but here you already have your design matrix X.
            ### YOUR CODE HERE
            w = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(trainX.T, trainX)), trainX.T),
                             traint)  # Calculate w vector (as an numpy.array)


            # This conditional returns until you implement the calculation of w
            if w is None:
                return None, None

            # calculate and record the mean squared losses

            train_pred = numpy.dot(trainX, w)  # model predictions on training data
            train_loss[fold, p] = numpy.mean(numpy.power(train_pred - traint, 2))

            fold_pred = numpy.dot(foldX, w)  # model predictions on held-out fold
            cv_loss[fold, p] = numpy.mean(numpy.power(fold_pred - foldt, 2))



    # The loss values can get quite large, so take the log for display purposes

    # Ensure taking log of the mean (not mean of the log!)
    mean_train_loss = numpy.mean(train_loss, 0)
    mean_cv_loss = numpy.mean(cv_loss, 0)

    log_mean_train_loss = numpy.log(mean_train_loss)
    log_mean_cv_loss = numpy.log(mean_cv_loss)

    print('\n----------------------\nResults for {0}'.format(title))
    print('log_mean_train_loss:\n{0}'.format(log_mean_train_loss))
    print('log_mean_cv_loss:\n{0}'.format(log_mean_cv_loss))

    min_mean_log_cv_loss = min(log_mean_cv_loss)
    # TODO: has to be better way to get the min index...
    best_poly = [i for i, j in enumerate(log_mean_cv_loss) if j == min_mean_log_cv_loss][0]

    best_fit_poly_X = numpy.zeros((x.shape[0], best_poly+1))

    for q in range(best_poly+1):
        best_fit_poly_X[:, q]= numpy.power(x, q)

    best_fit_model_w = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(best_fit_poly_X.T, best_fit_poly_X)), best_fit_poly_X.T), t)

    print('minimum mean_log_cv_loss of {0} for order {1}'.format(min_mean_log_cv_loss, best_poly))

    # Plot log scale loss results
    plot_cv_results(log_mean_train_loss, log_mean_cv_loss, ind_loss = None,
                    log_scale_p=True, plot_title=title)

    # Uncomment to plot direct-scale mean loss results
    #plot_cv_results(mean_train_loss, mean_cv_loss, mean_ind_loss, log_scale_p=False)

    return best_poly, min_mean_log_cv_loss, best_fit_model_w



# -----------------------------------------------------------------------------
# 5-fold cross validation
# -----------------------------------------------------------------------------

def exercise_3_5fold(DATA_SYNTH2019, PATH_TO_SYNTH_5FOLDCV_FIG, seed):
    # Load the data
    data_path = DATA_SYNTH2019
    figure_path = PATH_TO_SYNTH_5FOLDCV_FIG
    data = read_data(data_path, ',')
    x = data[:, 0]  # extract x (slice first column)
    t = data[:, 1]  # extract t (slice second column)

    numpy.random.seed(seed=seed)

    ### YOUR CODE HERE
    # Perform 5-fold cross-validation:
    K = 5
    k5_best_poly, k5_best_CVtest_log_MSE_loss, k5_best_model_w = run_cv_solution(K, 7, x, t, randomize_data=True, title='5-fold cross-validation')
    # Calculate the best polynomial model order as k5_best_poly
    # Calculate the log MSE 5-fold cv loss for the best polynomial model
    #       order on the synthetic data
    # Report the best-fit parameters w for teh best model order


    # NOTE: You can write a separate function that computes the
    #       best polynomial model order, the CV test log MSE loss for that
    #       model order, and the best-fit parameters for that model
    #       order, returning all three of these values.
    #       This has the advantage that you could reuse the same function
    #       in exercise_3_LOOCV. But this is NOT required.
    #       IF you do this, and supposed you called this function
    #       run_cv_solution() and it returns those three values, then
    #       you can set them as the return value of the function as
    #       follows:
    #k5_best_poly, k5_best_CVtest_log_MSE_loss, k5_best_model_w = run_cv_solution(K, x, t, randomize_data=True, title='5-fold cross-validation')

    #       rather than individually, as above.

    print('\n===== 5-fold CV')
    print(f'best model order: {k5_best_poly}')
    print(f'best model log MSE CV Test loss: {k5_best_CVtest_log_MSE_loss}')
    print(f'best model parameters: {k5_best_model_w}')
    plt.savefig(figure_path)
    return k5_best_poly, k5_best_CVtest_log_MSE_loss, k5_best_model_w


# -----------------------------------------------------------------------------
# Leave-one-out cross validation (LOOCV)
# -----------------------------------------------------------------------------

def exercise_3_LOOCV(DATA_SYNTH2019, PATH_TO_SYNTH_LOOCV_FIG, seed):
    # Load the data
    data_path = DATA_SYNTH2019
    data = read_data(data_path, ',')
    figure_path = PATH_TO_SYNTH_LOOCV_FIG
    x = data[:, 0]  # extract x (slice first column)
    t = data[:, 1]  # extract t (slice second column)

    numpy.random.seed(seed=seed)
    K=25

    ### YOUR CODE HERE
    # Perform Leave-One-Out cross-validation (LOOCV):
    # Calculate the best polynomial model order as loocv_best_model_order
    #loocv_best_model_order =
    # Calculate the log MSE CV test loss for the best polynomial model
    #       order on the synthetic data
    #loocv_best_CVtest_log_MSE_loss =
    # Report the best-fit parameters w for teh best model order
    #loocv_best_model_w =
    # NOTE: As noted in exercise_3_5fold, if you make a single function
    #       to return all three values, you can set all three variables
    #       at once, such as:
    loocv_best_model_order, loocv_best_CVtest_log_MSE_loss, loocv_best_model_w = run_cv_solution(K, 7, x, t, randomize_data=True, title='LOOCV')

    print('\n===== LOOCV')
    print(f'best model order: {loocv_best_model_order}')
    print(f'best model log MSE CV Test loss: {loocv_best_CVtest_log_MSE_loss}')
    print(f'best model parameters: {loocv_best_model_w}')
    plt.savefig(figure_path)

    return loocv_best_model_order, loocv_best_CVtest_log_MSE_loss, loocv_best_model_w


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_DEMO:
        run_demo(seed=29)
        plt.show()
    if RUN_EXERCISE_3_5FOLD:
        exercise_3_5fold(DATA_SYNTH2019, PATH_TO_SYNTH_5FOLDCV_FIG, seed=29)
        plt.show()
    if RUN_EXERCISE_3_LOOCV:
        exercise_3_LOOCV(DATA_SYNTH2019, PATH_TO_SYNTH_LOOCV_FIG, seed=29)
        plt.show()
