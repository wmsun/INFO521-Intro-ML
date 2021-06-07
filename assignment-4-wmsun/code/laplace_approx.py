import os
import numpy
import scipy.stats
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set the following to True in order to make the script execute
# the exercise function

RUN_LAPLACE_APPROX = True


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

FIGURES_ROOT = '../figures'
PATH_TO_LAPLACE1_FIG = os.path.join(FIGURES_ROOT, 'laplace_case1.png')
PATH_TO_LAPLACE2_FIG = os.path.join(FIGURES_ROOT, 'laplace_case2.png')
PATH_TO_LAPLACE3_FIG = os.path.join(FIGURES_ROOT, 'laplace_case3.png')


# -----------------------------------------------------------------------------
# Laplace approximation to the posterior Beta
# -----------------------------------------------------------------------------

def plot_laplace_approx(alpha, beta, N, y, figure_path):
    """
    Calculate an plot the posterior Beta distribution and Laplace approximation to that
    distribution given the prior Beta belief (alpha, beta) and observations (N, y).
    :param alpha: Prior beta alpha (or 'a' in scipy) parameter
    :param beta: Prior beta beta (or 'b' in scipy) parameter
    :param N: Number of coin flips
    :param y: Number of heads
    :return:
    """

    # create the x-axis values
    x = numpy.linspace(0., 1., 100)

    ### YOUR CODE HERE
    gamma = y+alpha  # posterior 'a' or 'alpha' parameter for the beta distribution
    delta = beta+N-y  # posterior 'b' or 'beta'  parameter for the beta distribution
    mu = (alpha+y-1)/(N+alpha+beta-2)     # the 'loc' (location) or 'mean' parameter for the Laplace approximation
    sigma_1 = mu*mu*(1-mu)*(1-mu)
    sigma_2 = (alpha+y-1)*(1-mu)*(1-mu)+(N+beta-y-1)*mu*mu
    sigma = numpy.sqrt(sigma_1/sigma_2)  # the 'scale' or 'standard deviation' parameter for the Laplace approximation

    ### DO NOT ALTER ANYTHING AFTER THIS LINE

    if gamma != 0 and delta != 0:
        beta_rv = scipy.stats.beta(a=gamma, b=delta)
        p = beta_rv.pdf(x)
    if mu != 0 and sigma != 0:
        gauss_rv = scipy.stats.norm(loc=mu, scale=sigma)
        q = gauss_rv.pdf(x)

    plt.figure()
    if gamma != 0 and delta != 0:
        plt.plot(x, p, label=r'$Beta(\gamma={0:.0f},\delta={1:.0f})$'.format(gamma, delta))
    if mu != 0 and sigma != 0:
        plt.plot(x, q, label=r'$\mathcal{N}' + r'({0:.2f},{1:.4f})$'.format(mu, sigma))
    plt.title(r'Laplace Approximation: $\alpha={0}$, $\beta={1}$, $N={2}$, $y={3}$'.format(alpha, beta, N, y))
    plt.xlabel(r'$r$')
    plt.ylabel(r'$p(r|y)$')
    plt.legend(loc="upper right")
    plt.savefig(figure_path, fmt='png')

    return gamma, delta, mu, sigma


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    if RUN_LAPLACE_APPROX:
        plot_laplace_approx(alpha=5, beta=5, N=20, y=10, figure_path=PATH_TO_LAPLACE1_FIG)
        plot_laplace_approx(alpha=3, beta=15, N=10, y=3, figure_path=PATH_TO_LAPLACE2_FIG)
        plot_laplace_approx(alpha=1, beta=30, N=10, y=3, figure_path=PATH_TO_LAPLACE3_FIG)
        plt.show()
