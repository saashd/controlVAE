import torch
import torchvision
import matplotlib.pyplot as plt

from Aux_functions import plot_figure
from Image_Generation.controlVAE import controlVAE
from dataLoader import load_mnist


def reconstruct_images(images, model):
    model.eval()
    with torch.no_grad():
        images, _, _ = model(images)
        images = images.clamp(0, 1)
        return images


def display_images(images,title):
    plt.figure(figsize=(17, 17))
    plt.imshow(torchvision.utils.make_grid(images[1:50], 10, 5).permute(1, 2, 0))
    plt.title(title, fontsize=50)
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', dpi=600)
    plt.show()


def main():
    controlVAE_180 = controlVAE(2, 1)
    betaVAE = controlVAE(2, 1)
    controlVAE_180.load_state_dict(torch.load('controlVAE_180_model', map_location=torch.device('cpu')))
    betaVAE.load_state_dict(torch.load('betaVAE_100_model', map_location=torch.device('cpu')))

    train_data = load_mnist(200)
    images, labels = iter(train_data).next()
    betaVAE_recon = reconstruct_images(images, betaVAE)
    controlVAE_180_recon = reconstruct_images(images, controlVAE_180)

    display_images(images,'Original MNIST Input')
    display_images(betaVAE_recon, "betaVAE-100  Output")
    display_images(controlVAE_180_recon,' controlVAE-180 Output')

    files = [
        'MNIST_output_betaVAE_100',
        'MNIST_output_controlVAE_170',
        'MNIST_output_controlVAE_180',
        'MNIST_output_controlVAE_200',
        'MNIST_output_VAE_0'
    ]
    plot_figure(files, 'recon_loss_list')
    plot_figure(files, 'kl_list')


if __name__ == '__main__':
    main()
