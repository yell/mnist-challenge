import os.path
import numpy as np


def import_trace(module_path, main_package_name,
                 include_main_package=True, discard_underscore_packages=True):
    """Return string representing sequence of imports needed
    to import module located in `module_path` from root of
    `main_package_name`.

    Examples
    --------
    For the following package structure:
    a
    - b
      - d.py
    - c
    Then `import_trace` ('.../a/b/d.py', 'a') will return 'a.b.d'.

    >>> import_trace(__file__, 'ml1_mnist')
    'ml1_mnist.utils'

    Raises
    ------
    ValueError
        If `main_package_name` is not present in `module_path`.
    """
    trace = ''
    head = module_path
    while True:
        head, tail = os.path.split(head)
        tail = os.path.splitext(tail)[0]
        if discard_underscore_packages and tail.startswith('_'):
            continue
        if not tail:
            raise ValueError("main package name '{0}' is not a part of '{1}'"\
                             .format(main_package_name, module_path))
        if tail == main_package_name:
            if include_main_package:
                trace = '.'.join(filter(bool, [tail, trace]))
            return trace
        trace = '.'.join(filter(bool, [tail, trace]))


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)