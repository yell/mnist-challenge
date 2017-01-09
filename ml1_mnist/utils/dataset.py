import struct
import os.path
import numpy as np

try:
    import seaborn as sns
    sns.set()
    from matplotlib import pyplot as plt
except ImportError:
    pass


def load_mnist(mode='train', path='.'):
    """
    Load and return MNIST dataset.

    Returns
    -------
    data : (n_samples, 784) ndarray
        Data representing raw pixel intensities (in 0.-255. range).
    target : (n_samples,) ndarray
        Labels vector.
    """

    if mode == 'train':
        fname_data = os.path.join(path, 'train-images.idx3-ubyte')
        fname_target = os.path.join(path, 'train-labels.idx1-ubyte')
    elif mode == 'test':
        fname_data = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_target = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("`mode` must be 'test' or 'train'")

    with open(fname_data, 'rb') as fdata:
        magic, n_samples, n_rows, n_cols = struct.unpack(">IIII", fdata.read(16))
        data = np.fromfile(fdata, dtype=np.uint8)
        data = data.reshape(n_samples, n_rows * n_cols)

    with open(fname_target, 'rb') as ftarget:
        magic, n_samples = struct.unpack(">II", ftarget.read(8))
        target = np.fromfile(ftarget, dtype=np.int8)

    return data.astype(float), target


def plot_mnist_digit(x, target=None):
    """Render a given array of pixel data."""
    image = x.reshape((28, 28))
    if 'sns' in globals():
        fig = plt.figure(figsize=(6, 5))
        ticklabels = range(28)
        ticklabels[::-2] = [''] * len(ticklabels[::-2])
        ax = sns.heatmap(image, cmap='Greys_r',
                         yticklabels=ticklabels, xticklabels=ticklabels)
        if target:
            ax.set_title("Label is {0}".format(target), fontsize=18)
        return ax
    else:
        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        if target:
            plt.title("Label is {0}".format(target), fontsize=18)