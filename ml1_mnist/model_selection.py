import numpy as np

from utils import RNG


class TrainTestSplitter(object):
    """
    A generic class for splitting data into (random) subsets.

    Parameters
    ----------
    shuffle : bool, optional
        Whether to shuffle the data.
    random_seed : None or int, optional
        Pseudo-random number generator seed used for random sampling.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 1, 2, 2, 3, 3, 3])

    >>> tts1 = TrainTestSplitter(shuffle=False)
    >>> train, test = tts1.split(y, train_ratio=0.5)
    >>> print y[train], y[test]
    [1 1 2] [2 3 3 3]
    >>> train, test = tts1.split(y, train_ratio=0.5, stratify=True)
    >>> print y[train], y[test]
    [1 2 3] [1 2 3 3]
    >>> for fold in tts1.make_k_folds(y, n_folds=3):
    ...     print y[fold]
    [1 1 2]
    [2 3]
    [3 3]
    >>> for fold in tts1.make_k_folds(y, n_folds=3, stratify=True):
    ...     print y[fold]
    [1 2 3]
    [1 2 3]
    [3]
    >>> for train, test in tts1.k_fold_split(y, n_folds=3):
    ...     print y[train], y[test]
    [2 3 3 3] [1 1 2]
    [1 1 2 3 3] [2 3]
    [1 1 2 2 3] [3 3]
    >>> for train, test in tts1.k_fold_split(y, n_folds=3, stratify=True):
    ...     print y[train], y[test]
    [1 2 3 3] [1 2 3]
    [1 2 3 3] [1 2 3]
    [1 2 3 1 2 3] [3]

    >>> tts2 = TrainTestSplitter(shuffle=True, random_seed=1337)
    >>> train, test = tts2.split(y, train_ratio=0.5)
    >>> print y[train], y[test]
    [3 2 1] [2 1 3 3]
    >>> train, test = tts2.split(y, train_ratio=0.5, stratify=True)
    >>> print y[train], y[test]
    [3 1 2] [3 3 2 1]
    >>> for fold in tts2.make_k_folds(y, n_folds=3):
    ...     print y[fold]
    [3 2 1]
    [2 1]
    [3 3]
    >>> for fold in tts2.make_k_folds(y, n_folds=3, stratify=True):
    ...     print y[fold]
    [3 1 2]
    [3 2 1]
    [3]
    """
    def __init__(self, shuffle=False, random_seed=None):
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.rng = RNG(self.random_seed)

    def split(self, y, train_ratio=0.8, stratify=False):
        """
        Split data into train and test subsets.

        Parameters
        ----------
        y : (n_samples,) array-like
            The target variable for supervised learning problems.
        train_ratio : float, 0 < `train_ratio` < 1, optional
            the proportion of the dataset to include in the train split.
        stratify : bool, optional
            If True, the folds are made by preserving the percentage of samples
            for each class. Stratification is done based upon the `y` labels.

        Returns
        -------
        train : (n_train,) ndarray
            The training set indices for that split.
        test : (n_samples - n_train,) ndarray
            The testing set indices for that split.
        """
        self.rng.reseed()
        n = len(y)

        if not stratify:
            indices = self.rng.permutation(n) if self.shuffle else np.arange(n, dtype=np.int)
            train_size = int(train_ratio * n)
            return np.split(indices, (train_size,))

        # group indices by label
        labels_indices = {}
        for index, label in enumerate(y):
            if not label in labels_indices: labels_indices[label] = []
            labels_indices[label].append(index)

        train, test = np.empty(0, dtype=np.int), np.empty(0, dtype=np.int)
        for label, indices in sorted(labels_indices.items()):
            size = int(train_ratio * len(indices))
            train = np.append(train, indices[:size])
            test  = np.append( test, indices[size:])

        if self.shuffle:
            self.rng.shuffle(train)
            self.rng.shuffle(test)

        return train, test

    def make_k_folds(self, y, n_folds=3, stratify=False):
        """
        Split data into folds of (approximately) equal size.

        Parameters
        ----------
        y : (n_samples,) array-like
            The target variable for supervised learning problems.
            Stratification is done based upon the `y` labels.
        n_folds : int, `n_folds` > 1, optional
            Number of folds.
        stratify : bool, optional
            If True, the folds are made by preserving the percentage of samples
            for each class. Stratification is done based upon the `y` labels.

        Yields
        ------
        fold : ndarray
            Indices for current fold.
        """
        self.rng.reseed()
        n = len(y)

        if not stratify:
            indices = self.rng.permutation(n) if self.shuffle else np.arange(n, dtype=np.int)
            for fold in np.array_split(indices, n_folds):
                yield fold
            return

        # group indices
        labels_indices = {}
        for index, label in enumerate(y):
            if not label in labels_indices:
                labels_indices[label] = []
            labels_indices[label].append(index)

        # split all indices label-wisely
        for label, indices in sorted(labels_indices.items()):
            labels_indices[label] = np.array_split(indices, n_folds)

        # collect respective splits into folds and shuffle if needed
        for k in xrange(n_folds):
            fold = np.concatenate([indices[k] for _, indices in sorted(labels_indices.items())])
            if self.shuffle:
                self.rng.shuffle(fold)
            yield fold

    def k_fold_split(self, y, n_folds=3, stratify=False):
        """
        Split data into train and test subsets for K-fold CV.

        Parameters
        ----------
        y : (n_samples,) array-like
            The target variable for supervised learning problems.
            Stratification is done based upon the `y` labels.
        n_folds : int, `n_folds` > 1, optional
            Number of folds.
        stratify : bool, optional
            If True, the folds are made by preserving the percentage of samples
            for each class. Stratification is done based upon the `y` labels.

        Yields
        ------
        train : (n_train,) ndarray
            The training set indices for current split.
        test : (n_samples - n_train,) ndarray
            The testing set indices for current split.
        """
        folds = list(self.make_k_folds(y, n_folds=n_folds, stratify=stratify))
        for i in xrange(n_folds):
            yield np.concatenate(folds[:i] + folds[(i + 1):]), folds[i]


if __name__ == '__main__':
    # run corresponding tests
    import test_model_selection as t
    from utils.testing import run_tests
    run_tests(__file__, t)