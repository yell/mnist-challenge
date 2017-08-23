import numpy as np
from unittest import skip

from ml1_mnist.model_selection import TrainTestSplitter as TTS


class TestSplit(object):
    def setUp(self):
        self.y = np.array([1, 1, 1, 2, 2, 3, 3,
                           1, 1, 2, 2, 2, 3, 3,
                           1, 1, 2, 2, 3, 3, 3])

    def test_split_no_shuffle(self):
        """Ensure order is preserved if no shuffle."""
        tts = TTS(shuffle=False)
        for train_ratio in np.arange(0.15, 0.96, 0.1):
            train, test = tts.split(self.y, train_ratio=train_ratio, stratify=False)
            np.testing.assert_allclose(self.y, np.concatenate((self.y[train], self.y[test])))

    @skip("use this test for `random.Random` rng")
    def test_split_random_seed(self):
        """
        Ensure for fixed (known) `random_seed` TTS instances produce
        the same outputs (for the same inputs).
        """
        for random_seed, stratify, (train_ref, test_ref) in (
            (42, False, (
                    np.array([6, 8, 9, 15, 7, 3, 17, 14, 11, 16]),
                    np.array([2, 19, 18, 1, 20, 10, 12, 4, 5, 0, 13])
            )),
            (1337, False, (
                np.array([ 8, 20,  9, 18,  1,  4,  0,  7, 14, 16]),
                np.array([17,  3, 15, 11,  5, 13,  2, 19,  6, 10, 12])
            )),
            (42, True, (
                    np.array([12, 6, 4, 2, 3, 5, 1, 0, 9]),
                    np.array([20, 13, 16, 17, 18, 15, 8, 19, 10, 14, 7, 11])
            )),
            (1337, True, (
                    np.array([6, 5, 1, 12, 0, 3, 2, 4, 9]),
                    np.array([14, 19, 17, 13, 10, 7, 16, 11, 8, 18, 15, 20])
            )),
        ):
            tts = TTS(shuffle=True, random_seed=random_seed)
            train, test = tts.split(self.y, train_ratio=0.5, stratify=stratify)
            np.testing.assert_allclose(train_ref, train)
            np.testing.assert_allclose(test_ref, test)

    def test_split_random_seed_2(self):
        """
        Ensure different TTS instances with the same `random_seed` produce
        the same outputs (for the same inputs).
        """
        for train_ratio in (0.25, 0.5, 0.75):
            for random_seed in np.random.randint(0, 1337, 25):
                for stratify in (False, True):
                    for i in xrange(2):
                        if i == 0:
                            tts1 = TTS(shuffle=True, random_seed=random_seed)
                            train1, test1 = tts1.split(self.y, train_ratio=train_ratio, stratify=stratify)
                        if i == 1:
                            tts2 = TTS(shuffle=True, random_seed=random_seed)
                            train2, test2 = tts2.split(self.y, train_ratio=train_ratio, stratify=stratify)
                            np.testing.assert_allclose(train1, train1)
                            np.testing.assert_allclose(test1, test1)

    def test_split_random_seed_3(self):
        """
        Ensure same TTS instance always produce
        the same outputs (for the same inputs).
        """
        for train_ratio in (0.25, 0.5, 0.75):
            for random_seed in np.random.randint(0, 1337, 25):
                for stratify in (False, True):
                    for i in xrange(2):
                        if i == 0:
                            tts = TTS(shuffle=True, random_seed=random_seed)
                            train1, test1 = tts.split(self.y, train_ratio=train_ratio, stratify=stratify)
                        if i == 1:
                            train2, test2 = tts.split(self.y, train_ratio=train_ratio, stratify=stratify)
                            np.testing.assert_allclose(train1, train1)
                            np.testing.assert_allclose(test1, test1)

    def test_split_stratification_no_shuffle(self):
        """Ensure stratification is preserved if no shuffle."""
        tts = TTS(shuffle=False)
        train, test = tts.split(self.y, train_ratio=4./7., stratify=True)
        assert np.count_nonzero(train == 1) ==\
               np.count_nonzero(train == 2) == np.count_nonzero(train == 3)
        assert np.count_nonzero(test == 1) == \
               np.count_nonzero(test == 2) == np.count_nonzero(test == 3)

    def test_split_stratification_random(self):
        """Ensure stratification is preserved even with shuffling."""
        for random_seed in np.random.randint(0, 1337, 100):
            tts = TTS(shuffle=True, random_seed=random_seed)
            train, test = tts.split(self.y, train_ratio=4./7., stratify=True)
            assert np.count_nonzero(train == 1) == \
                   np.count_nonzero(train == 2) == np.count_nonzero(train == 3)
            assert np.count_nonzero(test == 1) == \
                   np.count_nonzero(test == 2) == np.count_nonzero(test == 3)


    def test_make_k_folds_no_shuffle(self):
        """Ensure order is preserved if no shuffle (for different number of folds)."""
        for n_folds in xrange(2, 100):
            tts = TTS(shuffle=False)
            folds = list(tts.make_k_folds(self.y, n_folds=n_folds, stratify=False))
            np.testing.assert_allclose(np.arange(len(self.y)), np.concatenate(folds))

    def test_make_k_folds_stratification_no_shuffle(self):
        """Ensure stratification is preserved if no shuffle."""
        tts = TTS(shuffle=False)
        for fold in tts.make_k_folds(self.y, n_folds=7, stratify=True):
            np.testing.assert_allclose(np.sort(self.y[fold]), np.array([1, 2, 3]))

    def test_make_k_folds_stratification_random(self):
        """Ensure stratification is preserved even with shuffling."""
        for random_seed in np.random.randint(0, 1337, 100):
            tts = TTS(shuffle=True, random_seed=random_seed)
            for fold in tts.make_k_folds(self.y, n_folds=7, stratify=True):
                np.testing.assert_allclose(np.sort(self.y[fold]), np.array([1, 2, 3]))