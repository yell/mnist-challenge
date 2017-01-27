import os.path
import numpy as np

try:
    import seaborn as sns
    # sns.set()
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


def plot_learning_curves(l, a, vl, va, last_epochs=64, dirpath='.'):
    n_batches = len(l[0])
    n_epochs = len(l)
    x = np.linspace(1., n_epochs, n_epochs, endpoint=True)
    z = np.linspace(1., n_epochs, (n_epochs - 1) * n_batches, endpoint=True)
    # l = l[-last_epochs:]
    # a = a[-last_epochs:]
    # vl = vl[-last_epochs:]
    # va = va[-last_epochs:]
    # x = x[-last_epochs:]
    # z = z[-((last_epochs - 1) * n_batches):]

    plt.close("all")

    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax2 = ax.twinx()

    yl = ax.get_ygridlines()
    for yl_ in yl:
        yl_.set_color('w')

    l_mean = [np.mean(l_) for l_ in l]
    L1 = ax.plot(z, np.concatenate(l[1:]), color='#5a053f', lw=2, label='training loss')
    # ax.plot(x, l_mean, color='r', lw=2, marker='o', label='training loss mean')
    L2 = ax.plot(x, vl, color='#e6155a', lw=2, marker='o', label='validation loss')
    ax.set_ylim([0., max(1.0, max(max(l[1:])), max(vl))])

    L3 = ax2.plot(x, a, color='#124f90', lw=2, marker='o', label='training accuracy')
    L4 = ax2.plot(x, va, color='#6dbb30', lw=2, marker='o', label='validation accuracy')
    ax2.set_ylim([min(0.7, min(a), min(va)), 1.])

    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_color('black')
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['bottom'].set_linewidth(2)

    ax.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)

    ax.set_title('Learning curves', fontsize=27, weight='bold', y=1.03)

    ax.set_ylabel('loss', fontsize=23)
    ax2.set_ylabel('accuracy', fontsize=23)
    ax.set_xlabel('epoch', fontsize=23)

    Ls = L1 + L2 + L3 + L4
    labs = [l.get_label() for l in Ls]
    leg = plt.legend(Ls, labs, loc='lower left', fontsize=18, frameon=True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(2)

    fname = 'epoch_{0}.png'.format(n_epochs)
    fname = os.path.join(dirpath, fname)
    fig.savefig(fname, dpi=fig.dpi)
    plt.close(fig)


if __name__ == '__main__':
    from dataset import load_mnist
    X, y = load_mnist(mode='train', path='../../data/')
    plot_greyscale_image(X[0]/255., title='Label is {0}'.format(y[0]))
    plt.show()