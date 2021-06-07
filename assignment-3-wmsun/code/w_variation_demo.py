# w_variation_demo.py
# Port of w_variation_demo.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# The bias in the estimate of the variance
# Generate lots of datasets and look at how the average fitted variance
# agrees with the theoretical value
import numpy as np
# NOTE: If you are running on MacOSX and encounter an error like the following:
#     RuntimeError: Python is not installed as a framework.
#     The Mac OS X backend will not be able to function correctly
#     if Python is not installed as a framework...
# Then uncomment the following 2 lines:
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Generate the data sets and fit the parameters
true_w = np.array([-2, 3])  # The true model
N_dataset_sizes = np.linspace(20, 1000, 50)  # data set sizes
N_experiment_repetitions = 10000  # number of fit/predict repetitions

# Store sum-squared error for each experiment repetition (columns)
# at each data set size (rows)
all_ss = np.zeros((N_dataset_sizes.shape[0], N_experiment_repetitions))

# The true noise variance
noisevar = 0.5**2

# Total number of experiments at different data set sizes
total = N_dataset_sizes.shape[0]

print('NOTE: this will take a bit of time to run... generating', total, 'datasets')

# Generate experiments
for j in range(total):
    N = int(N_dataset_sizes[j])  # Number of observations in this data set
    print('processing data set', j+1, '(of', total, '), for data set of size', N)
    x = np.random.rand(N)
    X = np.zeros((x.shape[0], 2))
    X[:, 0] = np.power(x, 0)
    X[:, 1] = np.power(x, 1)
    for i in range(N_experiment_repetitions):
        t = np.dot(X, true_w) + np.random.randn(N)*np.sqrt(noisevar)
        w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, t))
        ss = (1.0 / N)*(np.dot(t, t) - np.dot(t, np.dot(X, w)))
        all_ss[j, i] = ss


# The expected value of the fitted variance is equal to:
# $\sigma^2\left(1-\frac{D}{N}\right)$
# where $D$ is the number of dimensions (2) and $\sigma^2$ is the true
# variance.
# Plot the average empirical value of the variance against the 
# theoretical expected value as the size of the datasets increases
plt.figure()
plt.scatter(N_dataset_sizes, np.mean(all_ss, 1), color='white', s=40,
            edgecolor='black', label='Empirical')
plt.plot(N_dataset_sizes, noisevar*(1-2.0/N_dataset_sizes), color='r', linewidth=2, label='Theoretical')
plt.plot(N_dataset_sizes, [0.25]*len(N_dataset_sizes),
         color='b', linestyle='dashed', label='Actual')
plt.xlabel('$N$ = number of samples')
plt.ylabel(r'$\mathbf{E}\{\widehat{\sigma^2}\}$')
plt.legend(loc=4)

# uncomment the following line to save the plot
# plt.savefig('w_variance_demo.png', fmt='png')

plt.pause(.1)  # required on some systems so that rendering can happen

plt.show()
