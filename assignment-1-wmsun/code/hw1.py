import numpy
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_EXERCISE_6 = True
RUN_EXERCISE_7 = False
RUN_EXERCISE_8 = False
RUN_EXERCISE_9 = False


# -----------------------------------------------------------------------------
# Global paths
# -----------------------------------------------------------------------------

DATA_ROOT = 'data'
PATH_TO_WALK_DATA = os.path.join(DATA_ROOT, 'walk.txt')
PATH_TO_X_DATA = os.path.join(DATA_ROOT, 'X.txt')
PATH_TO_W_DATA = os.path.join(DATA_ROOT, 'w.txt')

FIGURES_ROOT = 'figures'
PATH_TO_WALK_FIGURE = os.path.join(FIGURES_ROOT, 'walk.png')


# -----------------------------------------------------------------------------

def exercise_6(path_to_data, path_to_figure):
    """
    Plotting and scaling data
    :param path_to_data:
    :param path_to_figure:
    :return: result
    """

    print('='*30)
    print('Running exercise_6()')

    #### YOUR CODE HERE ####
    walk_arr = numpy.loadtxt('data/walk.txt')

    #### YOUR CODE HERE ####
    # plot the data using matplotlib plot!
    plt.plot(walk_arr)
    plt.ylabel('Location')
    plt.xlabel('Step')
    plt.title('Random Walk')
    plt.savefig('figures/walk.png')


    print(f'walk_arr.shape: {walk_arr.shape}')

    #### YOUR CODE HERE ####
    walk_min = numpy.amin(walk_arr)
    print(f'walk_min: {walk_min}')
    #### YOUR CODE HERE ####
    walk_max = numpy.amax(walk_arr)

    print(f'walk_max: {walk_max}')

    #### YOUR CODE HERE ####
    def scale01(arr):
        """
        linearly scale the values of an array in the range [0, 1]
        :param a: input ndarray
        :return: scaled ndarray
        """
        walk_arr_01 = numpy.interp(arr, (numpy.amin(arr), numpy.amax(arr)), (-1, +1)) # linear scaling
        return walk_arr_01 #return the scaled array

    walk_arr_scaled = scale01(walk_arr)


    print('DONE exercise_6()')

    return walk_arr, walk_min, walk_max, walk_arr_scaled


# -----------------------------------------------------------------------------

def exercise_7():
    """
    This is a doc string
    :return: results
    """
    print('=' * 30)
    print('Running exercise_7()')

    #### YOUR CODE HERE ####
    numpy.random.seed(7)
    # set the numpy random seed to 7

    # This determines how many times we "throw" the
    #   2 six-sided dice in an experiment
    num_dice_throws = 10000  # don't edit this!

    # This determines how many trials in each experiment
    #   ... that is, how many times we'll throw our two
    #   6-sided dice num_dice_throws times
    num_trials = 10  # don't edit this!

    # Yes, you can have functions inside of functions!
    def run_experiment():
        trial_outcomes = list()
        for trial in range(num_trials):
            #### YOUR CODE HERE ####
            doubles = 0 # number of the occurrence of doubles in one trial
            for throws in range(num_dice_throws):
                throw_0 = 1  # number of throws in one trial
                outcome = numpy.random.randint(1,7, size=2)  #generate two throws
                if outcome[0] == outcome[1]:
                    doubles = doubles + 1
                    #count the number of doubles
            # In the following, make it so that probability_estimate is an estimate
            # of the probability of throwing 'doubles' with two fair six-sided dice
            # (i.e., the probability that the dice end up with teh same values)
            # based on throwing the two dice num_dice_throws times.
            probability_estimate = doubles/num_dice_throws
            # Save the probability estimate for each trial (you don't need to change
            # this next line)
            trial_outcomes.append(probability_estimate)
            trial = trial + 1
        return trial_outcomes

    experiment_outcomes_1 = run_experiment()

    print(f'experiment_outcomes_1: {experiment_outcomes_1}')

    print(f'do it again!')

    experiment_outcomes_2 = run_experiment()
    print(f'experiment_outcomes_2: {experiment_outcomes_2}')

    print('Now reset the seed')

    #### YOUR CODE HERE ####
    numpy.random.seed(7)
    # reset the numpy random seed back to 7

    experiment_outcomes_3 = run_experiment()

    print(f'experiment_outcomes_3: {experiment_outcomes_3}')

    print("DONE exercise_7()")

    return experiment_outcomes_1, experiment_outcomes_2, experiment_outcomes_3


# -----------------------------------------------------------------------------

def exercise_8():
    """
    Random vectors and matrices, and some linear algebra operations
    :return: results
    """

    print("=" * 30)
    print("Running exercise_8()")

    #### YOUR CODE HERE ####
    numpy.random.seed(seed= 7)
    # set the numpy random seed to 7

    #### YOUR CODE HERE ####
    # Set x to a 2-d array of random number of shape (3, 1)
    x = numpy.random.rand(3, 1)

    print(f'x:\n{x}')

    #### YOUR CODE HERE ####
    # Set 7 to a 2-d array of random number of shape (3, 1)
    y = numpy.random.rand(3,1)

    print(f'y:\n{y}')

    #### YOUR CODE HERE ####
    # Calclate the sum of x and y
    v1 = x + y

    print(f'v1:\n{v1}')

    #### YOUR CODE HERE ####
    # Calclate the sum of x and y
    v2 = numpy.multiply(x, y)

    print(f'v2:\n{v2}')

    #### YOUR CODE HERE ####
    # Transpose x
    xT = numpy.transpose(x)

    print(f'xT: {xT}')

    #### YOUR CODE HERE ####
    # Calculate the dot product of x and y
    v3 = numpy.dot(xT, y)

    print(f'v3: {v3}')

    #### YOUR CODE HERE ####
    # Set A to a 2-d array of random numbers of shape (3, 3)
    A = numpy.random.rand(3,3)

    print(f'A:\n{A}')

    #### YOUR CODE HERE ####
    # Compute the dot product of x-transpose with A
    v4 = numpy.dot(xT, A)

    print(f'v4: {v4}')

    #### YOUR CODE HERE ####
    # Compute the dot product of x-transpose with A and the product with y
    v5 = numpy.dot(v4, y)

    print(f'v5: {v5}')

    #### YOUR CODE HERE ####
    # Compute the inverse of A
    v6 = numpy.linalg.inv(A)

    print(f'v6:\n{v6}')

    #### YOUR CODE HERE ####
    # Compute the dot product of A with its inverse.
    #   Should be near identity (save for some numerical error)
    v7 = numpy.dot(v6, A)

    print(f'v7:\n{v7}')

    return x, y, v1, v2, xT, v3, A, v4, v5, v6, v7


# -----------------------------------------------------------------------------

def exercise_9(path_to_X_data, path_to_w_data):
    """
    Implementing scalar versus vector math
    :param path_to_X_data:
    :param path_to_w_data:
    :return:
    """

    print("="*30)
    print("Running exercise_9()")

    #### YOUR CODE HERE ####
    # load the X and w data from file into arrays
    X = numpy.loadtxt('data/X.txt', delimiter=',')
    w = numpy.loadtxt('data/w.txt', delimiter=',')

    print(f'X:\n{X}')
    print(f'w: {w}')

    #### YOUR CODE HERE ####
    # Extract the column 0 (x_n1) and column 1 (x_n2) vectors from X

    x_n1 = X[numpy.array([0,1,2,3,4]), 0]
    x_n2 = X[numpy.array([0,1,2,3,4]), 1]

    print(f'x_n1: {x_n1}')
    print(f'x_n2: {x_n2}')

    #### YOUR CODE HERE ####
    w_0 = w[0]
    w_1 = w[1]

    scalar_result_0 = w_0 * w_0 * sum(x_n1*x_n1) + 2 * w_0 * w_1 * sum(x_n2 * x_n1) + w_1 * w_1 * sum(x_n2*x_n2)
    # Use scalar arithmetic to compute the right-hand side of Exercise 3
    #   (Exercise 1.3 from FCMA p.35)
    # Set the final value to
    scalar_result = scalar_result_0

    print(f'scalar_result: {scalar_result}')

    #### YOUR CODE HERE ####
    # Now you will compute the same result but using linear algebra operators.
    #   (i.e., the left-hand of the equation in Exercise 1.3 from FCMA p.35)
    # You can compute the values in any linear order you want (but remember,
    # linear algebra is *NOT* commutative!), however here will require you to
    # first compute the inner term: X-transpose times X (XX), and then
    # below you complete the computation by multiplying on the left and right
    # by w (wXXw)
    X_transpose = numpy.transpose(X)
    XX = numpy.dot(X_transpose, X)

    print(f'XX:\n{XX}')

    #### YOUR CODE HERE ####
    # Now you'll complete the computation by multiplying on the left and right
    # by w to determine the final value: wXXw
    wXX = numpy.dot(w, XX)
    wXXw = numpy.dot(wXX, w)

    print(f'wXXw: {wXXw}')

    print("DONE exercise_9()")

    return X, w, x_n1, x_n2, scalar_result, XX, wXXw


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_EXERCISE_6:
        exercise_6(PATH_TO_WALK_DATA, PATH_TO_WALK_FIGURE)
        #### YOUR CODE HERE ####
        plt.show()

        # Add a call to the matplotlib.pyplot show() function
    if RUN_EXERCISE_7:
        exercise_7()
    if RUN_EXERCISE_8:
        exercise_8()
    if RUN_EXERCISE_9:
        exercise_9(PATH_TO_X_DATA, PATH_TO_W_DATA)
