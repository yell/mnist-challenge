import os.path
import struct
import numpy as np
from matplotlib import pyplot as plt


def load_mnist(mode='train', path='.'):
    """
    Load and return MNIST dataset.

    Returns
    -------
    data : (n_samples, 784) ndarray
        data representing raw pixel intensities (0-255)
    target : (n_samples,) ndarray
        labels vector
    """

    if mode == 'train':
        fname_data = os.path.join(path, 'train-images.idx3-ubyte')
        fname_target = os.path.join(path, 'train-labels.idx1-ubyte')
    elif mode == 'test':
        fname_data = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_target = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("`mode` must be 'test' | 'train'")

    with open(fname_data, 'rb') as fdata:
        magic, n_samples, n_rows, n_cols = struct.unpack(">IIII", fdata.read(16))
        data = np.fromfile(fdata, dtype=np.uint8).reshape(n_samples, n_rows * n_cols)

    with open(fname_target, 'rb') as ftarget:
        magic, n_samples = struct.unpack(">II", ftarget.read(8))
        target = np.fromfile(ftarget, dtype=np.int8)

    return data, target


def plot_mnist_digit(x, target=None):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    image = x.reshape((28, 28))
    plt.imshow(image, cmap='gray')
    if target:
        plt.title("Label is {0}".format(target))