import numpy
import os


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

DATA_ROOT = os.path.join('..', 'data')
PATH_TO_RANDOM_TEST = os.path.join(DATA_ROOT, 'rand_test.txt')


# -----------------------------------------------------------------------------
# RNG Class
# -----------------------------------------------------------------------------

class RNG:
    """
    Simple class to emulate a random number generator using pre-computed
    random number sequence (either provided in file or directly as sequence
    in constructor).
    """
    def __init__(self, source_file, random_numbers=None):
        """
        Seed RNG either by loading file of random numbers in source_file
        or directly from random_numbers sequence (when source_file is None)
        :param source_file: path to source file (if None, fill using
         argument random_numbers)
        :param random_numbers: (default None) sequence
        """
        if source_file is not None:
            self.random_numbers = numpy.loadtxt(source_file)
        else:
            self.random_numbers = numpy.array(random_numbers)
        self.counter = 0
        self.total = self.random_numbers.shape[0] - 1

    def get_1_random(self):
        """
        Return one random scalar value.
        :return: scalar value
        """
        if self.counter > self.total:
            self.counter = 0
        v = self.random_numbers[self.counter]
        self.counter += 1
        return v

    def get_n_random(self, n: int):
        """
        Return next n random values. Returns numpy array of n values.
        :param n: number of values to draw
        :return: numpy array of n values
        """
        if n < 0:
            raise Exception(f'ERROR RNG.get_n_random(): argument n = {n}; MUST be integer value >= 0.')
        if self.counter + n > self.total:
            counter = self.counter
            new_n = (counter + n) - self.total
            self.counter = 0
            return numpy.concatenate((self.random_numbers[counter:], self.get_n_random(new_n)))
        else:
            start = self.counter
            self.counter += n
            return self.random_numbers[start: self.counter]

    def stats(self):
        print(f'size: {self.random_numbers.shape[0]}')
        print(f'min:  {numpy.min(self.random_numbers)}')
        print(f'max:  {numpy.max(self.random_numbers)}')
        print(f'mean: {numpy.mean(self.random_numbers)}')


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def save_random_01(filepath, size):
    numpy.savetxt(filepath, numpy.random.random(size))


def save_random_uniform_ab(filepath, low, high, size):
    numpy.savetxt(filepath, numpy.random.uniform(low, high, size))


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_circular(verbose_p=False):
    if verbose_p:
        print('START test_circular()')
    rng = RNG(None, numpy.array(list(range(10))))
    rng.total = rng.random_numbers.shape[0]
    if verbose_p:
        print(f'rng.random_numbers:   {rng.random_numbers}')
        print(f'rng.total:            {rng.total}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(rng.random_numbers, numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert rng.total == 10
    assert rng.counter == 0
    r = rng.get_n_random(6)
    if verbose_p:
        print(f'rng.get_n_random(6):  {r.shape[0]} {r}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(r, numpy.array([0, 1, 2, 3, 4, 5]))
    assert rng.counter == 6
    r = rng.get_n_random(6)
    if verbose_p:
        print(f'rng.get_n_random(6):  {r.shape[0]} {r}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(r, numpy.array([6, 7, 8, 9, 0, 1]))
    assert rng.counter == 2
    r = rng.get_n_random(22)
    if verbose_p:
        print(f'rng.get_n_random(22): {r.shape[0]} {r}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(r, numpy.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]))
    assert rng.counter == 4
    r = rng.get_n_random(3)
    if verbose_p:
        print(f'rng.get_n_random(3):  {r.shape[0]} {r}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(r, numpy.array([4, 5, 6]))
    assert rng.counter == 7
    r = rng.get_n_random(3)
    if verbose_p:
        print(f'rng.get_n_random(3):  {r.shape[0]} {r}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(r, numpy.array([7, 8, 9]))
    assert rng.counter == 10
    r = rng.get_n_random(0)
    if verbose_p:
        print(f'rng.get_n_random(0):  {r.shape[0]} {r}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(r, numpy.array([]))
    assert rng.counter == 10
    r = rng.get_n_random(3)
    if verbose_p:
        print(f'rng.get_n_random(3):  {r.shape[0]} {r}')
        print(f'rng.counter:          {rng.counter}')
    assert numpy.array_equal(r, numpy.array([0, 1, 2]))
    assert rng.counter == 3
    if verbose_p:
        print('DONE test_circular()')


test_circular()


def scale_lh(x, low, high):
    return (x * (high - low)) + low


def test1():
    save_random_01(PATH_TO_RANDOM_TEST, 10)
    rng = RNG(PATH_TO_RANDOM_TEST)
    rng.stats()


def test2():
    save_random_uniform_ab(PATH_TO_RANDOM_TEST, -4, 10, 10)
    rng = RNG(PATH_TO_RANDOM_TEST)
    rng.stats()
    for i in range(30):
        print('  ', i, rng.get_1_random())
    print('DONE.')


def test_scale_lh():
    for v in numpy.linspace(0, 1, 10):
        print(v, scale_lh(v, -4, 10))


# test_scale_lh()


def test3():
    save_random_01(PATH_TO_RANDOM_TEST, 10)
    rng = RNG(PATH_TO_RANDOM_TEST)
    rng.stats()
    for i in range(30):
        r = rng.get_1_random()
        print('  ', i, r, scale_lh(r, -4, 10))


#test3()
