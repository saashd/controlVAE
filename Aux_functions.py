import pickle
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt


def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose((npimg * 255).astype('uint8'), (1, 2, 0)))
    plt.title(title)
    plt.show()


def plot_output(files, output_type):
    for file_name in files:
        file = open(file_name, 'rb')
        object_file = pickle.load(file)
        file.close()
        x = [i for i in range(len(object_file[output_type]))]
        y = object_file[output_type]
        from scipy.ndimage.filters import gaussian_filter1d
        ysmoothed = gaussian_filter1d(y, sigma=2)
        plt.plot(x, ysmoothed, label=file_name)


def plot_figure(files, output_type):
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_output(files, output_type)
    plt.tick_params(labelsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    if output_type == 'kl_list':
        title = 'KL Divergence'
        limit = [0, 300]
        plt.axhline(y=170, color='black')
        plt.axhline(y=180, color='black')
        plt.axhline(y=200, color='black')
    elif output_type == 'recon_loss_list':
        title = 'Reconstruction Loss'
        limit = [0, 4000]
    plt.ylabel(title, fontsize=15)
    plt.ylim(limit)
    plt.xlim(0, 6000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x))))
    plt.legend(loc='best', prop={'size': 11.5})
    plt.grid()
    plt.tight_layout()
    fig.savefig(output_type, bbox_inches='tight', dpi=600)
    plt.show()
