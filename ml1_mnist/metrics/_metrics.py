# TODO: add validation routine (numpyify if needed + check equal lengths)
import numpy as np


def accuracy_score(y_true, y_pred, normalize=True):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : (n_samples, n_outputs) array-like
        Ground truth (correct) labels.
    y_pred : (n_samples, n_outputs) array-like
        Predicted labels, as returned by a classifier.
    normalize : bool, optional
        If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    Returns
    -------
    score : float or int
        If `normalize` == True, return the correctly classified
        samples (float), else return the number of correctly
        classified samples (int).
    """
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    score = sum(np.all(z == t) for z, t in zip(y_pred, y_true))
    if normalize:
        score /= float(len(y_true))
    return score


def zero_one_loss(y_true, y_pred, normalize=True):
    """Zero-one classification loss.

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
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    loss = sum(np.any(z != t) for z, t in zip(y_pred, y_true))
    if normalize:
        loss /= float(len(y_true))
    return loss


# aliases
misclassification_rate = zero_one_loss


if __name__ == '__main__':
    # run corresponding tests
    import test_metrics as t
    import env; from utils.testing import run_tests
    run_tests(__file__, t)