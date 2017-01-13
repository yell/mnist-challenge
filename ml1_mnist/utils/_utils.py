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
    >>> np.abs(s.elapsed_time - 0.1) < 0.01
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
            self.elapsed_time = None

    def __enter__(self, verbose=False):
        self.start = self.timerfunc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = self.timerfunc() - self.start
        if self.verbose:
            print "Elapsed time: {0:.3f} sec".format(self.elapsed_time)


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)