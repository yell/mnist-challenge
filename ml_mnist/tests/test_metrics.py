import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises

from ml_mnist.metrics import get_metric


class TestAccuracyScore(object):
    def __init__(self):
        self.f = get_metric('accuracy_score')

    def test_equal(self):
        y_true, y_pred = [1, 2, 3, 4], [1, 2, 3, 4]
        assert_almost_equal(self.f(y_true, y_pred), 1.)
        assert self.f(y_true, y_pred, normalize=False) == 4

    def test_not_equal(self):
        y_true, y_pred = [2, 2, 3, 4], [1, 2, 3, 4]
        assert_almost_equal(self.f(y_true, y_pred), 0.75)
        assert self.f(y_true, y_pred, normalize=False) == 3

    def test_str_equal(self):
        y_true, y_pred = ['1', '2', '3', '4'], ['1', '2', '3', '4']
        assert_almost_equal(self.f(y_true, y_pred), 1.)
        assert self.f(y_true, y_pred, normalize=False) == 4

    def test_str_not_equal(self):
        y_true, y_pred = ['2', '2', '3', '4'], ['1', '2', '3', '4']
        assert_almost_equal(self.f(y_true, y_pred), 0.75)
        assert self.f(y_true, y_pred, normalize=False) == 3

    def test_multilabel_equal(self):
        y_true, y_pred = [[0, 1], [0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]]
        assert_almost_equal(self.f(y_true, y_pred), 1.)
        assert self.f(y_true, y_pred, normalize=False) == 3

    def test_multilabel_not_equal(self):
        y_true, y_pred = np.array([[0, 1], [1, 1], [0, 0]]), np.ones((3, 2))
        assert_almost_equal(self.f(y_true, y_pred), 1./3.)
        assert self.f(y_true, y_pred, normalize=False) == 1

    def test_empty(self):
        y_true, y_pred = [], []
        assert_raises(Exception, self.f, y_true, y_pred)


class TestZeroOneLoss(object):
    def __init__(self):
        self.f = get_metric('zero_one_loss')

    def test_equal(self):
        y_true, y_pred = [1, 2, 3, 4], [1, 2, 3, 4]
        assert_almost_equal(self.f(y_true, y_pred), 0.)
        assert self.f(y_true, y_pred, normalize=False) == 0

    def test_not_equal(self):
        y_true, y_pred = [2, 2, 3, 4], [1, 2, 3, 4]
        assert_almost_equal(self.f(y_true, y_pred), 0.25)
        assert self.f(y_true, y_pred, normalize=False) == 1

    def test_str_equal(self):
        y_true, y_pred = ['1', '2', '3', '4'], ['1', '2', '3', '4']
        assert_almost_equal(self.f(y_true, y_pred), 0.)
        assert self.f(y_true, y_pred, normalize=False) == 0

    def test_str_not_equal(self):
        y_true, y_pred = ['2', '2', '3', '4'], ['1', '2', '3', '4']
        assert_almost_equal(self.f(y_true, y_pred), 0.25)
        assert self.f(y_true, y_pred, normalize=False) == 1

    def test_multilabel_equal(self):
        y_true, y_pred = [[0, 1], [0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]]
        assert_almost_equal(self.f(y_true, y_pred), 0.)
        assert self.f(y_true, y_pred, normalize=False) == 0

    def test_multilabel_not_equal(self):
        y_true, y_pred = np.array([[0, 1], [1, 1], [0, 0]]), np.ones((3, 2))
        assert_almost_equal(self.f(y_true, y_pred), 2./3.)
        assert self.f(y_true, y_pred, normalize=False) == 2

    def test_empty(self):
        y_true, y_pred = [], []
        assert_raises(Exception, self.f, y_true, y_pred)


class TestLogLoss(object):
    def __init__(self):
        self.f = get_metric('log_loss')

    def test_equal(self):
        assert_almost_equal(self.f([1.], [1.]), 0.)
        assert_almost_equal(self.f([0.], [0.]), 0.)
        assert_almost_equal(self.f([0., 1.], [0., 1.]), 0.)
        assert_almost_equal(self.f([1., 0.], [1., 0.]), 0.)

    def test_not_equal(self):
        assert_almost_equal(self.f([1.], [0.5]), -np.log(0.5))
        assert_almost_equal(self.f([0., 1.], [0.5, 0.5]), -0.5 * np.log(0.5))
