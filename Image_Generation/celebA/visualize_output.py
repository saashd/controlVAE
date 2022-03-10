import torch
from Aux_functions import plot_figure, reconstruct_celebA, display_celebA
from Image_Generation.controlVAE import controlVAE
from dataLoader import load_celeba


def main():
    VAE = controlVAE(500, 3)
    VAE.load_state_dict(torch.load('VAE_0_model', map_location=torch.device('cpu')))

    controlVAE_170 = controlVAE(500, 3)
    controlVAE_170.load_state_dict(torch.load('controlVAE_170_model', map_location=torch.device('cpu')))

    betaVAE_100 = controlVAE(500, 3)
    betaVAE_100.load_state_dict(torch.load('betaVAE_100_model', map_location=torch.device('cpu')))

    controlVAE_180 = controlVAE(500, 3)
    controlVAE_180.load_state_dict(torch.load('controlVAE_180_model', map_location=torch.device('cpu')))

    controlVAE_200 = controlVAE(500, 3)
    controlVAE_200.load_state_dict(torch.load('controlVAE_200_model', map_location=torch.device('cpu')))

    train_data = load_celeba(30)
    images = iter(train_data).next()

    VAE_recon = reconstruct_celebA(images, VAE)
    controlVAE_170_recon = reconstruct_celebA(images, controlVAE_170)
    betaVAE_100_recon = reconstruct_celebA(images, betaVAE_100)
    controlVAE_180_recon = reconstruct_celebA(images, controlVAE_180)
    controlVAE_200_recon = reconstruct_celebA(images, controlVAE_200)

    display_celebA(images, 'Original CelebA Input')
    display_celebA(VAE_recon, ' VAE Output')
    display_celebA(controlVAE_170_recon, 'controlVAE-170 Output')
    display_celebA(betaVAE_100_recon, 'betaVAE-100 Output')
    display_celebA(controlVAE_180_recon, "controlVAE-180  Output")
    display_celebA(controlVAE_200_recon, "controlVAE-200  Output")

    files = [
        'celebA_output_betaVAE_100',
        'celebA_output_controlVAE_170',
        'celebA_output_controlVAE_180',
        'celebA_output_controlVAE_200',
        'celebA_output_VAE_0'
    ]
    plot_figure(files, 'recon_loss_list')
    plot_figure(files, 'kl_list')


if __name__ == '__main__':
    main()
