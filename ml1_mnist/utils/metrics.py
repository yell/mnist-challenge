import numpy as np


def zero_one_loss(y_true, y_pred, normalize=True):
    """
    Zero-one classification loss.

    Parameters
    ----------
    y_true : (n_samples, n_outputs) array-like
        Ground truth (correct) labels.
    y_pred : (n_samples, n_outputs) array-like
        Predicted labels, as returned by a classifier.
    normalize : bool, optional
        If False, return number of misclassifications.
        Otherwise, return the fraction of misclassifications.

    Returns
    -------
    loss : float or int
        If `normalize` == True, return the fraction of
        misclassifications (float), else it returns
        the number of misclassifications (int).
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    loss = sum(np.any(z != t) for z, t in zip(y_pred, y_true))
    if normalize:
        loss /= float(max(len(y_true), 1))
    return loss


# aliases
misclassification_rate = zero_one_loss


if __name__ == '__main__':
    # run corresponding tests
    import tests.test_metrics as t
    from testing import run_tests
    run_tests(__file__, t)