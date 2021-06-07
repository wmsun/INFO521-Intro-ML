
# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------
import numpy
# Set the following to True in order to make the script execute
# the exercise function

RUN_PI_ESTIMATE = True


# -----------------------------------------------------------------------------
# Estimate pi
# -----------------------------------------------------------------------------

def estimate_pi(N):
    """
    Calculate an estimate of pi (i.e., DO NOT use math.pi or any other direct
    representation of pi) by sampling, when only provided the ability to sample
    from a uniform distribution (using random.uniform) and the fact that area,
    A, of a circle with radius, r, is defined by: A = pi * r^2
    :param N: Number of samples
    :return: estimate of pi
    """

    ### YOUR CODE HERE
    inside = 0
    for i in range(0, N):
        x_0 = numpy.random.uniform()**2
        y_0 = numpy.random.uniform()**2
        if  numpy.sqrt(x_0 + y_0)< 1.0:
            inside += 1

    estimated_pi = inside/N *4 # Calculate pi! pi != 0
    print(estimated_pi)

    return estimated_pi


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_PI_ESTIMATE:
        estimate_pi(1000000)

