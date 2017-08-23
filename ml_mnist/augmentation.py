import numpy as np
import scipy.ndimage.interpolation
import scipy.ndimage.filters

try:
    import seaborn as sns
    sns.set()
    from matplotlib import pyplot as plt
except ImportError:
    pass

from utils import RNG, plot_greyscale_image
from utils.dataset import load_mnist



def get_transformation(transformation_name, **params):
    for k, v in globals().items():
        if k.lower() == transformation_name.lower():
            return v(**params)
    raise ValueError("invalid transformation name '{0}'".format(transformation_name))



class RandomTransformation(object):
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.rng = RNG(self.random_seed)

    def __call__(self, x):
        self.rng = RNG(self.random_seed)
        return self._call(x)

    def _call(self, x):
        raise NotImplementedError()


def shift(x, shift_=(0, 0)):
    y = scipy.ndimage.interpolation.shift(x, shift=shift_, mode='nearest')
    y = (y - y.min()) * (x.max() - x.min()) / (y.max() - y.min()) + x.min()
    return y


class RandomShift(RandomTransformation):
    def __init__(self, x_shift=(0, 0), y_shift=(0, 0), random_seed=None):
        self.x_shift = x_shift
        self.y_shift = y_shift
        super(RandomShift, self).__init__(random_seed=random_seed)

    def _call(self, x):
        x_shift = self.rng.randint(self.x_shift[0], self.x_shift[1] + 1)
        y_shift = self.rng.randint(self.y_shift[0], self.y_shift[1] + 1)
        return shift(x, shift_=(y_shift, x_shift))


def rotate(x, angle=0.):
    y = scipy.ndimage.interpolation.rotate(x, angle=angle, mode='nearest', reshape=False)
    y = (y - y.min()) * (x.max() - x.min()) / (y.max() - y.min()) + x.min()
    return y


class RandomRotate(RandomTransformation):
    def __init__(self, angle=(0., 0.), random_seed=None):
        self.angle = angle
        super(RandomRotate, self).__init__(random_seed=random_seed)

    def _call(self, x):
        angle = self.rng.uniform(self.angle[0], self.angle[1])
        return rotate(x, angle=angle)


def subsample(x, pos=(0, 0), new_shape=None):
    new_shape = new_shape or x.shape
    y = x[pos[0]:(pos[0] + new_shape[0]), pos[1]:(pos[1] + new_shape[1])]
    return np.copy(y)


class RandomSubsample(RandomTransformation):
    def __init__(self, new_shape=None, random_seed=None):
        self.new_shape = new_shape
        super(RandomSubsample, self).__init__(random_seed=random_seed)

    def _call(self, x):
        new_shape = self.new_shape or x.shape
        x_pos = self.rng.randint(0, x.shape[0] - new_shape[0] + 1)
        y_pos = self.rng.randint(0, x.shape[1] - new_shape[1] + 1)
        return subsample(x, pos=(x_pos, y_pos), new_shape=new_shape)


def gaussian(x, sigma=0.):
    y = scipy.ndimage.filters.gaussian_filter(x, sigma=sigma, mode='nearest')
    y = (y - y.min()) * (x.max() - x.min()) / (y.max() - y.min()) + x.min()
    return y


class RandomGaussian(RandomTransformation):
    def __init__(self, sigma=(0., 0.), random_seed=None):
        self.sigma = sigma
        super(RandomGaussian, self).__init__(random_seed=random_seed)

    def _call(self, x):
        sigma = self.rng.uniform(self.sigma[0], self.sigma[1])
        return gaussian(x, sigma)


class Dropout(RandomTransformation):
    def __init__(self, p=(0., 0.), random_seed=None):
        self.p = p
        super(Dropout, self).__init__(random_seed=random_seed)

    def _call(self, x):
        p = self.rng.uniform(self.p[0], self.p[1])
        mask = self.rng.uniform(size=x.shape) > p
        mask = mask * (x.max() - x.min()) + x.min()
        return np.minimum(x, mask)



class RandomAugmentator(RandomTransformation):
    def __init__(self, transform_shape=None, out_shape=None, random_seed=None):
        self.transform_shape = transform_shape
        self.out_shape = out_shape
        self.transforms = []
        self._inner_seed = None
        super(RandomAugmentator, self).__init__(random_seed=random_seed)

    def _update_inner_seed(self):
        self._inner_seed = self.rng.randint(2 ** 20, size=len(self.transforms))

    def add(self, transformation, **params):
        self.transforms.append(get_transformation(transformation, **params))
        return self

    def transform_x(self, x, n_samples=3):
        x = np.asarray(x)
        for _ in xrange(n_samples):
            y = np.copy(x)
            if self.transform_shape: y = y.reshape(self.transform_shape)
            self._update_inner_seed()
            for i, t in enumerate(self.transforms):
                t.random_seed = self._inner_seed[i]
                y = t(y)
            if self.out_shape: y = y.reshape(self.out_shape)
            yield y

    def transform(self, X, n_samples=3):
        X_new = []
        for x in X:
            X_new.append(x)
            for y in self.transform_x(x, n_samples=n_samples):
                if not self.out_shape:
                    y = y.reshape(x.shape)
                X_new.append(y)
        return np.asarray(X_new)


if __name__ == '__main__':
    X, y = load_mnist(mode='train', path='../data/')

    # x = X[0]/255.
    # y = shift(x, (-3, 3))
    # y = RandomShift(x_shift=(-5, 5), y_shift=(-5, 5), random_seed=1337)(x)
    # y = rotate(x, 30.)
    # y = RandomRotate((-30, 30), random_seed=1337)(x)
    # y = gaussian(x, 1.5) # 0 < 1.5
    # y = RandomGaussian((1., 2.), random_seed=1337)(x)
    # y = subsample(x, (5, 5), (16, 16))
    # y = RandomSubsample((20, 20), random_seed=1337)(x)
    # y = Dropout((0.9, 1.0), random_seed=np.array([23, 31, 98]))(x)

    aug = RandomAugmentator(transform_shape=(28, 28), random_seed=1337)
    aug.add('RandomRotate', angle=(-10., 15.))
    aug.add('Dropout', p=(0.9, 1.))
    aug.add('RandomGaussian', sigma=(0., 1.))
    aug.add('RandomShift', x_shift=(-2, 2), y_shift=(-2, 2))

    for y in aug.transform(X[:3]/255., 3):
        plot_greyscale_image(y)
        plt.show()