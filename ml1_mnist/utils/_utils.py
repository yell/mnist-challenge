import os.path
import numpy as np


def is_param_name(name):
    return not name.startswith('_') and not name.endswith('_')

def is_attribute_name(name):
    return not name.startswith('_') and name.endswith('_')

def is_param_or_attribute_name(name):
    return not name.startswith('_')


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


def pformat(params, offset, printer=repr):
    """Pretty format the dictionary `params`.

    Parameters
    ----------
    params : dict
        The dictionary to pretty print.
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable, optional
        The function to convert entries to strings, typically
        the builtin str or repr.

    Returns
    -------
    pformatted : str
        Pretty formatted `params`.
    """
    np_print_options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=32, edgeitems=2)

    params_strs = []
    current_line_len = offset
    line_sep = ',\n' + min(1 + offset / 2, 8) * ' '

    for key, value in sorted(params.items()):
        this_repr = "{0}={1}".format(key, printer(value))
        if len(this_repr) > 256:
            this_repr = this_repr[:192] + '...' + this_repr[-64:]
        if (current_line_len + len(this_repr) >= 75 or '\n' in this_repr):
            params_strs.append(line_sep)
            current_line_len = len(line_sep)
        elif params_strs:
            params_strs.append(', ')
            current_line_len += 2
        params_strs.append(this_repr)
        current_line_len += len(this_repr)

    np.set_printoptions(**np_print_options)

    pformatted = ''.join(params_strs)
    # strip trailing space to avoid nightmare in doctests
    pformatted = '\n'.join(l.rstrip() for l in pformatted.split('\n'))
    return pformatted


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)