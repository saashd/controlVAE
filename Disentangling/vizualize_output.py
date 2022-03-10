import torch
from torch.utils.data import Dataset

from Aux_functions import plot_output, reconstruct_MNIST, plot_figure, select_img_samples, show_images_grid
import random
from Disentangling.controlVAE import controlVAE
from dataLoader import load_dSprites

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    files = [
        'dSprites_output_betaVAE_100',
        'dSprites_output_controlVAE_16',
        'dSprites_output_controlVAE_18',
    ]
    plot_figure(files, 'recon_loss_list', 'dSprites')
    plot_figure(files, 'kl_list', 'dSprites')

    controlVAE_16 = controlVAE(10, 1)
    controlVAE_16.load_state_dict(torch.load('controlVAE_16_model', map_location=torch.device('cpu')))

    # controlVAE_18 = controlVAE(10, 1)
    # controlVAE_18.load_state_dict(torch.load('controlVAE_18_model', map_location=torch.device('cpu')))
    #
    # betaVAE_100 = controlVAE(10, 1)
    # betaVAE_100.load_state_dict(torch.load('betaVAE_100_model', map_location=torch.device('cpu')))

    train_data = load_dSprites(10000, '../data/dSprites/dsprites_subset.npz')
    images, labels = iter(train_data).next()

    idx = select_img_samples(labels, latent_variable=0)
    idx.sort()
    samples = min(len(idx), 10)
    indices = random.sample(idx, samples)
    rand_imgs = images[indices]
    rand_labels = labels[indices]
    show_images_grid(rand_imgs, rand_labels, 'controlVAE_16 Shape', controlVAE_16, samples)
    # show_images_grid(rand_imgs, rand_labels, 'controlVAE_18 Shape', controlVAE_18, samples)
    # show_images_grid(rand_imgs, rand_labels, 'betaVAE_100 Shape', betaVAE_100, samples)


if __name__ == '__main__':
    main()
