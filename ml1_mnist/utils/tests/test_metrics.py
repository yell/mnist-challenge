import numpy as np

import env
from metrics import zero_one_loss


class TestZeroOneLoss(object):

    def test_equal(self):
        y_true, y_pred = [1, 2, 3, 4], [1, 2, 3, 4]
        np.testing.assert_allclose(0., zero_one_loss(y_true, y_pred))
        assert zero_one_loss(y_true, y_pred, normalize=False) == 0

    def test_not_equal(self):
        y_true, y_pred = [2, 2, 3, 4], [1, 2, 3, 4]
        np.testing.assert_allclose(0.25, zero_one_loss(y_true, y_pred))
        assert zero_one_loss(y_true, y_pred, normalize=False) == 1

    def test_str_equal(self):
        y_true, y_pred = ['1', '2', '3', '4'], ['1', '2', '3', '4']
        np.testing.assert_allclose(0., zero_one_loss(y_true, y_pred))
        assert zero_one_loss(y_true, y_pred, normalize=False) == 0

    def test_str_not_equal(self):
        y_true, y_pred = ['2', '2', '3', '4'], ['1', '2', '3', '4']
        np.testing.assert_allclose(0.25, zero_one_loss(y_true, y_pred))
        assert zero_one_loss(y_true, y_pred, normalize=False) == 1

    def test_multilabel_equal(self):
        y_true, y_pred = [[0, 1], [0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]]
        np.testing.assert_allclose(0., zero_one_loss(y_true, y_pred))
        assert zero_one_loss(y_true, y_pred, normalize=False) == 0

    def test_multilabel_not_equal(self):
        y_true, y_pred = np.array([[0, 1], [1, 1], [0, 0]]), np.ones((3, 2))
        np.testing.assert_allclose(2./3., zero_one_loss(y_true, y_pred))
        assert zero_one_loss(y_true, y_pred, normalize=False) == 2

    def test_empty(self):
        y_true, y_pred = [], []
        np.testing.assert_allclose(0., zero_one_loss(y_true, y_pred))