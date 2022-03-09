import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from Aux_functions import plot_figure
from Image_Generation.controlVAE import controlVAE, reparametrization_trick
from dataLoader import load_mnist
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def reconstruct_images(images, model):
    model.eval()
    with torch.no_grad():
        images, _, _ = model(images)
        images = images.clamp(0, 1)
        return images


def display_images(images, title):
    plt.figure(figsize=(17, 17))
    plt.imshow(torchvision.utils.make_grid(images, 5, 10).permute(1, 2, 0))
    plt.title(title, fontsize=50)
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', dpi=600)
    plt.show()


def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
    _, mu, log_var = autoencoder.encoder(x_1.unsqueeze(0))
    z_1 = reparametrization_trick(mu, log_var)
    _, mu, log_var = autoencoder.encoder(x_2.unsqueeze(0))
    z_2 = reparametrization_trick(mu, log_var)

    z = torch.stack([z_1 + (z_2 - z_1) * t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy() * 255

    images_list = [Image.fromarray(img.reshape(128, 128)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1]  # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)


def main():
    files = [
        'MNIST_output_betaVAE_100',
        'MNIST_output_controlVAE_170',
        'MNIST_output_controlVAE_180',
        'MNIST_output_controlVAE_200',
        'MNIST_output_VAE_0'
    ]
    plot_figure(files, 'recon_loss_list')
    plot_figure(files, 'kl_list')

    VAE = controlVAE(2, 1)
    VAE.load_state_dict(torch.load('VAE_0_model', map_location=torch.device('cpu')))

    controlVAE_170 = controlVAE(2, 1)
    controlVAE_170.load_state_dict(torch.load('controlVAE_170_model', map_location=torch.device('cpu')))

    controlVAE_180 = controlVAE(2, 1)
    controlVAE_180.load_state_dict(torch.load('controlVAE_180_model', map_location=torch.device('cpu')))

    controlVAE_200 = controlVAE(2, 1)
    controlVAE_200.load_state_dict(torch.load('controlVAE_200_model', map_location=torch.device('cpu')))

    betaVAE_100 = controlVAE(2, 1)
    betaVAE_100.load_state_dict(torch.load('betaVAE_100_model', map_location=torch.device('cpu')))

    train_data = load_mnist(100)
    images, labels = iter(train_data).next()

    x_1 = images[labels == 1][1].to(device)  # find a 1
    x_2 = images[labels == 0][1].to(device)  # find a 0

    interpolate_gif(VAE, "vae", x_1, x_2)
    interpolate_gif(controlVAE_170, "controlVAE_170", x_1, x_2)
    interpolate_gif(controlVAE_180, "controlVAE_180", x_1, x_2)
    interpolate_gif(controlVAE_200, "controlVAE_200", x_1, x_2)
    interpolate_gif(betaVAE_100, "betaVAE_100", x_1, x_2)

    VAE_recon = reconstruct_images(images, VAE)
    controlVAE_170_recon = reconstruct_images(images, controlVAE_170)
    controlVAE_180_recon = reconstruct_images(images, controlVAE_180)
    controlVAE_200_recon = reconstruct_images(images, controlVAE_200)
    betaVAE_100_recon = reconstruct_images(images, betaVAE_100)

    display_images(images, 'Original MNIST Input')
    display_images(VAE_recon, ' VAE Output')
    display_images(controlVAE_170_recon, ' controlVAE-170 Output')
    display_images(controlVAE_180_recon, "controlVAE-180  Output")
    display_images(controlVAE_200_recon, "controlVAE-200  Output")
    display_images(betaVAE_100_recon, ' betaVAE-100 Output')


if __name__ == '__main__':
    main()
