import pickle
import numpy as np
import torch
import torchvision
from matplotlib import ticker
import matplotlib.pyplot as plt
from PIL import Image

from Image_Generation.controlVAE import reparametrization_trick


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


def plot_figure(files, output_type, data='celebA'):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_output(files, output_type)
    plt.tick_params(labelsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    title = ''
    limit = [0, 0]
    if output_type == 'kl_list':
        title = 'KL Divergence'
        if data in ['celebA', 'MNIST']:
            limit = [0, 300]
            plt.axhline(y=170, color='black')
            plt.axhline(y=180, color='black')
            plt.axhline(y=200, color='black')
        else:
            limit = [0, 50]
            plt.axhline(y=18, color='black')
            plt.axhline(y=16, color='black')

    elif output_type == 'recon_loss_list':
        title = 'Reconstruction Loss'
        if data == 'celebA':
            limit = [0, 3000]
        elif data == 'MNIST':
            limit = [0, 2000]
        else:
            limit = [0, 1000]
    plt.ylabel(title, fontsize=15)
    plt.ylim(limit)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x))))
    plt.legend(loc='best', prop={'size': 11.5})
    plt.grid()
    plt.tight_layout()
    fig.savefig(output_type, bbox_inches='tight', dpi=600)
    plt.show()


def reconstruct_celebA(images, model):
    model.eval()
    with torch.no_grad():
        images, _, _ = model(images)
        for i in range(len(images)):
            min_ele = torch.min(images[i])
            images[i] -= min_ele
            images[i] /= torch.max(images[i])
        return images


def reconstruct_MNIST(images, model):
    model.eval()
    with torch.no_grad():
        images, _, _ = model(images)
        images = images.clamp(0, 1)
        return images


def display_celebA(images, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(torchvision.utils.make_grid(images, 5, 10).permute(1, 2, 0))
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', dpi=600)
    plt.show()


def display_MNIST(images, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(torchvision.utils.make_grid(images, 5, 10).permute(1, 2, 0))
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', dpi=600)
    plt.show()


def interpolate_gif(autoencoder, filename, x_1, x_2, data='MNIST', n=100):
    _, mu, log_var = autoencoder.encoder(x_1.unsqueeze(0))
    z_1 = reparametrization_trick(mu, log_var)
    _, mu, log_var = autoencoder.encoder(x_2.unsqueeze(0))
    z_2 = reparametrization_trick(mu, log_var)

    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy() * 255
    if data == 'MNIST':
        images_list = [Image.fromarray(img.reshape(128, 128)).resize((256, 256)) for img in interpolate_list]
    else:
        images_list = [Image.fromarray(img.reshape(64, 64)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1]  # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)


def show_images_grid(rand_imgs, rand_labels, title, model, num_images=25):
    rand_reconst_imgs = reconstruct_MNIST(rand_imgs, model).reshape(-1, 64, 64)
    side_by_side = np.array(rand_imgs).reshape(-1, 64, 64)
    imgs_ = []
    for i in range(0, num_images):
        imgs_.append(side_by_side[i])
    for i in range(0, num_images):
        imgs_.append(rand_reconst_imgs[i])
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()
    for ax_i, ax in enumerate(axes):
        if ax_i < num_images * 2:
            ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    fig.tight_layout()
    plt.show()


def select_img_samples(labels, latent_variable=1):
    idx = []
    j = latent_variable
    for i, x in enumerate(labels):
        for k, label in enumerate(x):
            if j == 0 and 0.5 <= label[1].item() <= 1 and 3 <= label[2].item() <= 5 and \
                    0.5 <= label[3].item() <= 1 and 0.5 <= label[4].item() <= 1:
                idx.append(i)
            elif j == 1 and label[0].item() == 1 and 0 <= label[2].item() <= 1 and \
                    0.6 < label[3].item() <= 0.8 and 0.6 < label[4].item() <= 0.8:
                idx.append(i)
            elif j == 2 and label[0].item() == 2 and 0.0 < label[1].item() <= 0.5 and \
                    0.0 < label[4].item() <= 0.2 and 0.0 < label[3].item() <= 0.2:
                idx.append(i)
            elif j == 3 and label[0].item() == 1 and 0.0 < label[1].item() <= 0.5 and \
                    0.0 < label[2].item() <= 1 and 0.0 < label[4].item() <= 0.2:
                idx.append(i)
            elif j == 4 and label[0].item() == 1 and 0.0 < label[1].item() <= 0.5 and \
                    0.0 < label[2].item() <= 1 and 0.0 < label[3].item() <= 0.2:
                idx.append(i)
    return idx
