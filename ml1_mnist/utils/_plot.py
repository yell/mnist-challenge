import numpy as np

try:
    import seaborn as sns
    sns.set()
    from matplotlib import pyplot as plt
except ImportError:
    pass


def plot_greyscale_image(x, shape=(28, 28), title=None):
    """Render a given array of pixel data."""
    image = np.asarray(x).reshape(shape)
    if 'sns' in globals():
        fig = plt.figure(figsize=(6, 5))
        xticklabels = range(shape[1])
        xticklabels[::-2] = [''] * len(xticklabels[::-2])
        yticklabels = range(shape[0])
        yticklabels[::-2] = [''] * len(yticklabels[::-2])
        ax = sns.heatmap(image, cmap='Greys_r',
                         xticklabels=xticklabels,
                         yticklabels=yticklabels)
        if title:
            ax.set_title(title, fontsize=18)
        return ax
    else:
        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        if title:
            plt.title(title, fontsize=18)


if __name__ == '__main__':
    from dataset import load_mnist
    X, y = load_mnist(mode='train', path='../../data/')
    plot_greyscale_image(X[0]/255., title='Label is {0}'.format(y[0]))
    plt.show()