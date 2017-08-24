import sys
import os.path
import numpy as np
from copy import deepcopy

from utils import one_hot
from utils.read_write import save_model
from metrics import get_metric


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

    >>> import_trace(__file__, 'ml_mnist')
    'ml_mnist.base'

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
            raise ValueError("main package name '{0}' is not a part of '{1}'" \
                             .format(main_package_name, module_path))
        if tail == main_package_name:
            if include_main_package:
                trace = '.'.join(filter(bool, [tail, trace]))
            return trace
        trace = '.'.join(filter(bool, [tail, trace]))
    return trace

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


class BaseEstimator(object):
    """Base class for all estimators."""

    def __init__(self, _y_required=True):
        self._y_required = _y_required
        self._X = None
        self._y = None
        self._n_samples = None
        self._n_features = None
        self._n_outputs = None
        self._called_fit = False
        self._store_default_params()

    def _check_X_y(self, X, y=None):
        """
        Ensure inputs are in the expected format:

        Convert `X` [and `y`] to `np.ndarray` if needed
        and ensure `X` [and `y`] are not empty.

        Parameters
        ----------
        X : (n_samples, n_features) array-like
            Data (feature vectors).
        y : (n_samples,) or (n_samples, n_outputs) array-like
            Labels vector. By default is required, but may be omitted
            if `_y_required` is False.
        """
        # validate `X`
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.size == 0:
            raise ValueError('number of features must be > 0')

        if X.ndim == 1:
            self._n_samples, self._n_features = 1, X.shape
        else:
            self._n_samples, self._n_features = X.shape[0], np.prod(X.shape[1:])

        # validate `y` if needed
        if self._y_required:
            if y is None:
                raise ValueError('missed required argument `y`')

            if not isinstance(y, np.ndarray):
                y = np.asarray(y)

            if y.size == 0:
                raise ValueError('number of outputs must be > 0')

            # TODO: decide whether to check len(y) == self._n_samples (for semi-supervised learning)?
            if y.ndim == 1:
                self._n_outputs = 1
            else:
                self._n_outputs = np.prod(y.shape[1:])

        self._X = X
        self._y = y

    def _fit(self, X, y=None, **fit_params):
        """Class-specific `fit` routine."""
        raise NotImplementedError()

    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data."""
        self._check_X_y(X, y)
        if self._y_required:
            self._fit(self._X, self._y, **fit_params)
        else:
            self._fit(self._X, **fit_params)
        self._called_fit = True
        return self

    def _predict(self, X=None, **predict_params):
        """Class-specific `predict` routine."""
        raise NotImplementedError()

    def predict(self, X=None, **predict_params):
        """Predict the target for the provided data."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if self._called_fit:
            return self._predict(X, **predict_params)
        else:
            raise ValueError('`fit` must be called before calling `predict`')

    def evaluate(self, X, y_true, metric='accuracy_score'):
        y_pred = self.predict(X)
        if len(y_pred.shape) == 2 and y_pred.shape[1] > 1 and (len(y_true.shape) == 1 or y_true.shape[1] == 1):
            y_true = one_hot(y_true)
        return get_metric(metric)(y_true, y_pred)

    def model_name(self):
        return self.__class__.__name__

    def get_params(self, deep=True, **params_mask):
        """Get parameters (and attributes) of the model.

        Parameters
        ----------
        deep : bool, optional
            Whether to deepcopy all the parameters.
        params_mask : kwargs, optional
            Enables to control which parameters to include/exclude.
            If some parameters set to True, return only them.
            If some parameters set to False, return all excluding them.
            If there are mixed parameters, ValueError is raised.

        Returns
        -------
        params : dict
            Parameters of the model. Includes all members (not methods)
            and also model name being class name stored as 'model'.
        """
        if all(x in map(bool, params_mask.values()) for x in (False, True)):
            raise ValueError('`params_mask` cannot contain True and False values simultaneously')

        # collect all attributes
        params = vars(self)

        # omit "interesting" members
        params = {key: params[key] for key in params if is_param_or_attribute_name(key)}

        # filter according to the mask provided
        if params_mask:
            if params_mask.values()[0]:
                params = {key: params[key] for key in params if key in params_mask}
            else:
                params = {key: params[key] for key in params if not key in params_mask}

        # path where the actual classifier is stored
        module_name = sys.modules[self.__class__.__module__].__name__
        module_path = os.path.abspath(module_name.replace('.', '/'))
        trace = import_trace(
            module_path=module_path,
            main_package_name='ml_mnist',
            include_main_package=False
        )
        class_name = self.__class__.__name__
        params['model'] = '.'.join([trace, class_name])
        if deep:
            params = deepcopy(params)
        return params

    def _store_default_params(self):
        params = vars(self)
        params = {key: params[key] for key in params if is_param_name(key)}
        self._default_params = deepcopy(params)

    def reset_params(self):
        """Restore default params (that were passed to the constructor).

        Returns
        -------
        self
        """
        for key, value in self._default_params.items():
            setattr(self, key, value)
        return self

    def set_params(self, **params):
        """Set parameters (and attributes) of the model.

        Parameters
        ----------
        params : kwargs
            New parameters and their values.

        Returns
        -------
        self
        """
        for key, value in params.items():
            if is_param_or_attribute_name(key) and hasattr(self, key):
                setattr(self, key, value)
        return self

    def _serialize(self, params):
        """Class-specific parameters serialization routine."""
        return params

    def _deserialize(self, params):
        """Class-specific parameters deserialization routine."""
        return params

    def save(self, filepath=None, params_mask={}, json_params={}):
        save_model(self, filepath, params_mask, json_params)

    def __repr__(self):
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        del params['model']
        return "{0}({1})".format(class_name,
                                 pformat(params, offset=len(class_name)))


if __name__ == '__main__':
    # run corresponding tests
    import env
    from utils.testing import run_tests
    run_tests(__file__)
