# TODO: add validation routine (numpyify if needed + check equal lengths)
import numpy as np
try:
    import seaborn as sns
    sns.set()
    from matplotlib import pyplot as plt
except ImportError:
    pass


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
    score = sum(np.all(t == p) for t, p in zip(y_true, y_pred))
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
    loss = sum(np.any(t != p) for t, p in zip(y_true, y_pred))
    if normalize:
        loss /= float(len(y_true))
    return loss


def confusion_matrix(y_true, y_pred, labels=None, normalize=None):
    """Compute confusion matrix.

    By definition a confusion matrix C is such that C_{i, j}
    is equal to the number of observations known to be in group i
    but predicted to be in group j.

    Thus in binary classification, the count of
     true negatives is C_{0,0},
    false negatives is C_{1,0},
     true positives is C_{1,1} and
    false positives is C_{0,1}.

    Notes
    -----
    Only `n_outputs` = 1 case is supported for now.

    Parameters
    ----------
    y_true : (n_samples,) array-like
        Ground truth (correct) labels.
    y_pred : (n_samples,) array-like
        Predicted labels, as returned by a classifier.
    labels : (n_classes,) array-like
        List of labels to index the matrix. If no `labels`
        are provided, they are assumed to be [0, 1, ..., N],
        where N is max(max(`y_true`), max(`y_pred`)).
    normalize : None or {'rows', 'cols'}, optional
        Whether to normalize confusion matrix.
        If `normalize` == 'rows', normalize c.m. to have rows summed to 1,
        this is useful for finding how each class has been classified.
        If `normalize` == 'cols', normalize c.m. to have columns summed to 1,
        this is useful for finding what classes are responsible for each classification.

    Returns
    -------
    C : (n_classes, n_classes) ndarray
        Confusion matrix.

    Examples
    --------
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    >>> confusion_matrix(y_true, y_pred, labels=[0, 2])
    array([[2, 0],
           [1, 2]])
    >>> confusion_matrix(y_true, y_pred, labels=range(4))
    array([[2, 0, 0, 0],
           [0, 0, 1, 0],
           [1, 0, 2, 0],
           [0, 0, 0, 0]])
    >>> confusion_matrix(y_true, y_pred, normalize='rows')
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ],
           [ 0.33333333,  0.        ,  0.66666667]])
    >>> confusion_matrix(y_true, y_pred, normalize='cols')
    array([[ 0.66666667,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.33333333],
           [ 0.33333333,  0.        ,  0.66666667]])
    """
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    labels = labels or range(max(max(y_true), max(y_pred)) + 1)

    C = np.zeros((len(labels), len(labels)), dtype=np.int)
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            C[labels.index(t)][labels.index(p)] += 1

    if normalize == 'rows':
        row_sums = C.astype(np.float).sum(axis=1)[:, np.newaxis]
        C = C / np.maximum(np.ones_like(row_sums), row_sums)

    elif normalize == 'cols':
        col_sums = C.astype(np.float).sum(axis=0)
        C = C / np.maximum(np.ones_like(col_sums), col_sums)

    return C


def plot_confusion_matrix(C, labels=None, labels_fontsize=None, **heatmap_params):
    fig = plt.figure()

    # default params
    labels = labels or range(C.shape[0])
    labels_fontsize = labels_fontsize or 13
    annot_fontsize = 14
    xy_label_fontsize = 21

    # set default params where possible
    if not 'annot' in heatmap_params:
        heatmap_params['annot'] = True
    if not 'fmt' in heatmap_params:
        heatmap_params['fmt'] = 'd' if C.dtype is np.dtype('int') else '.3f'
    if not 'annot_kws' in heatmap_params:
        heatmap_params['annot_kws'] = {'size': annot_fontsize}
    elif not 'size' in heatmap_params['annot_kws']:
        heatmap_params['annot_kws']['size'] = annot_fontsize
    if not 'xticklabels' in heatmap_params:
        heatmap_params['xticklabels'] = labels
    if not 'yticklabels' in heatmap_params:
        heatmap_params['yticklabels'] = labels

    # plot the stuff
    with plt.rc_context(rc={'xtick.labelsize': labels_fontsize,
                            'ytick.labelsize': labels_fontsize}):
        ax = sns.heatmap(C, **heatmap_params)
        plt.xlabel('predicted', fontsize=xy_label_fontsize)
        plt.ylabel('actual', fontsize=xy_label_fontsize)
        return ax


# aliases
misclassification_rate = zero_one_loss


if __name__ == '__main__':
    # run corresponding tests
    import test_metrics as t
    import env; from utils.testing import run_tests
    run_tests(__file__, t)