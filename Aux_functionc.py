import numpy as np
import torch.nn.functional as F
from matplotlib import ticker
import matplotlib.pyplot as plt

def reconstruction_loss(x, x_recon):
    batch_size = x.size(0)
    x_recon = F.sigmoid(x_recon)
    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    return total_kld


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose((npimg * 255).astype('uint8'), (1, 2, 0)))
    plt.show()


def plot_figure(x, y, x_title, y_title, fig_title):
    fig, ax = plt.subplots()
    plt.plot(x, y)
    plt.tick_params(labelsize=15)
    plt.xlabel(x_title, fontsize=15)
    plt.ylabel(y_title, fontsize=15)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x / 100)) + 'K'))
    plt.legend(loc='best', prop={'size': 11.5})
    plt.grid()
    plt.tight_layout()
    fig.savefig(fig_title, bbox_inches='tight', dpi=600)
    plt.show()