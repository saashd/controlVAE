import torch
from Aux_functions import plot_figure, reconstruct_MNIST, interpolate_gif, display_MNIST
from Image_Generation.controlVAE import controlVAE
from dataLoader import load_mnist

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    VAE_recon = reconstruct_MNIST(images, VAE)
    controlVAE_170_recon = reconstruct_MNIST(images, controlVAE_170)
    controlVAE_180_recon = reconstruct_MNIST(images, controlVAE_180)
    controlVAE_200_recon = reconstruct_MNIST(images, controlVAE_200)
    betaVAE_100_recon = reconstruct_MNIST(images, betaVAE_100)

    display_MNIST(images, 'Original MNIST Input')
    display_MNIST(VAE_recon, ' VAE Output')
    display_MNIST(controlVAE_170_recon, ' controlVAE-170 Output')
    display_MNIST(controlVAE_180_recon, "controlVAE-180  Output")
    display_MNIST(controlVAE_200_recon, "controlVAE-200  Output")
    display_MNIST(betaVAE_100_recon, ' betaVAE-100 Output')


if __name__ == '__main__':
    main()
