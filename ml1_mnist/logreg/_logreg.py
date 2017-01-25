import numpy as np

import env
from base import BaseEstimator
from optimizers import get_optimizer


class LogisticRegression(BaseEstimator):
    """Multinomial Logistic Regression.

    Parameters
    ----------
    penalty : {'l1', 'l2'}, optional
        Used to specify the norm used in the penalization.
    C : float, optional
        Inverse of regularization strength; must be a positive float.
        Smaller values specify stronger regularization.
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call
        to fit as initialization, otherwise, just erase
        the previous solution.
    optimizer : {'adam'}, optional
        Specifies which optimizer to use in the algorithm.
    optimizer_params : kwargs
        Additional kwargs passed to `optimizer`
    """
    def __init__(self, penalty='l2', C=1.0, warm_start=True,
                 optimizer='adam', optimizer_params={}):
        self.warm_start = warm_start
        self.penalty = penalty
        self.C = C
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        super(LogisticRegression, self).__init__(_y_required=True)


    def _fit(self, X, y, verbose=False):
        # TODO: just use NN
        pass

    def _serialize(self, params):
        return params

    def _deserialize(self, params):
        return params