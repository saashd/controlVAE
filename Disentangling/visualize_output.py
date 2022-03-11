import torch
from Aux_functions import plot_figure, interpolate_gif
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

    controlVAE_18 = controlVAE(10, 1)
    controlVAE_18.load_state_dict(torch.load('controlVAE_18_model', map_location=torch.device('cpu')))

    betaVAE_100 = controlVAE(10, 1)
    betaVAE_100.load_state_dict(torch.load('betaVAE_100_model', map_location=torch.device('cpu')))

    train_data = load_dSprites(10000, '../data/dSprites/dsprites_subset.npz')
    images, labels = iter(train_data).next()

    x_1 = images[labels[:, 0, 0] == 1][1].to(device)
    x_2 = images[labels[:, 0, 0] == 3][1].to(device)
    interpolate_gif(controlVAE_16, "controlVAE_16 dsprites shape", x_1, x_2, 'dSprites')
    x_1 = images[labels[:, 0, 1] == 0.5][1].to(device)
    x_2 = images[labels[:, 0, 1] == 1.][1].to(device)
    interpolate_gif(controlVAE_16, "controlVAE_16 dsprites scale", x_1, x_2, 'dSprites')
    x_1 = images[labels[:, 0, 2] == 0.][1].to(device)
    x_2 = images[labels[:, 0, 2] == 0.][1].to(device)
    interpolate_gif(controlVAE_16, "controlVAE_16 dsprites Orientation", x_1, x_2, 'dSprites')
    x_1 = images[labels[:, 0, 3] == 0.][1].to(device)
    x_2 = images[labels[:, 0, 3] == 1.][1].to(device)
    interpolate_gif(controlVAE_16, "controlVAE_16 dsprites X-position", x_1, x_2, 'dSprites')
    x_1 = images[labels[:, 0, 4] == 0.][1].to(device)
    x_2 = images[labels[:, 0, 4] == 1.][1].to(device)
    interpolate_gif(controlVAE_16, "controlVAE_16 dsprites y-position", x_1, x_2, 'dSprites')


if __name__ == '__main__':
    main()
