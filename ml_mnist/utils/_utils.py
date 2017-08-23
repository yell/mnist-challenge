import sys
import time
import numpy as np


class Stopwatch(object):
    """
    Simple class encapsulating stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as s:
    ...     time.sleep(0.1) # doctest: +ELLIPSIS
    Elapsed time: 0.100... sec
    >>> with Stopwatch(verbose=False) as s:
    ...     time.sleep(0.1)
    >>> np.abs(s.elapsed() - 0.1) < 0.01
    True
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        if sys.platform == "win32":
            # on Windows, the best timer is time.clock()
            self.timerfunc = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self.timerfunc = time.time
        self.start_ = None
        self.elapsed_ = None

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.stop().elapsed()
        if self.verbose:
            print "Elapsed time: {0:.3f} sec".format(elapsed)

    def start(self):
        self.start_ = self.timerfunc()
        self.elapsed_ = None
        return self

    def stop(self):
        self.elapsed_ = self.timerfunc() - self.start_
        self.start_ = None
        return self

    def elapsed(self):
        if self.start_ is None:
            return self.elapsed_
        return self.timerfunc() - self.start_


def print_inline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def width_format(x, default_width=8, max_precision=3):
    len_int_x = len(str(int(x)))
    width = max(len_int_x, default_width)
    precision = min(max_precision, max(0, default_width - 1 - len_int_x))
    return "{0:{1}.{2}f}".format(x, width, precision)


def one_hot(y):
    """Convert `y` to one-hot encoding.

    Examples
    --------
    >>> y = [2, 1, 0, 2, 0]
    >>> one_hot(y)
    array([[ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]])
    """
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]


def one_hot_decision_function(y):
    """
    Examples
    --------
    >>> y = [[0.1, 0.4, 0.5],
    ...      [0.8, 0.1, 0.1],
    ...      [0.2, 0.2, 0.6],
    ...      [0.3, 0.4, 0.3]]
    >>> one_hot_decision_function(y)
    array([[ 0.,  0.,  1.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.]])
    """
    z = np.zeros_like(y)
    z[np.arange(len(z)), np.argmax(y, axis=1)] = 1
    return z


def unhot(y):
    """
    Map `y` from one-hot encoding to {0, ..., `n_classes` - 1}.

    Examples
    --------
    >>> y = [[0, 0, 1],
    ...      [0, 1, 0],
    ...      [1, 0, 0],
    ...      [0, 0, 1],
    ...      [1, 0, 0]]
    >>> unhot(y)
    array([2, 1, 0, 2, 0])
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    _, n_classes = y.shape
    return y.dot(np.arange(n_classes))


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)