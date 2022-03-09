import torch
import torchvision
import matplotlib.pyplot as plt

from Aux_functions import plot_figure
from Image_Generation.controlVAE import controlVAE
from dataLoader import load_celeba


def reconstruct_images(images, model):
    model.eval()

    with torch.no_grad():
        images, _, _ = model(images)
        for i in range(len(images)):
            min_ele = torch.min(images[i])
            images[i] -= min_ele
            images[i] /= torch.max(images[i])
        return images


def display_images(images, title):
    plt.figure(figsize=(17, 17))
    plt.imshow(torchvision.utils.make_grid(images[1:50], 10, 5).permute(1, 2, 0))
    plt.title(title, fontsize=50)
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', dpi=600)
    plt.show()


def main():
    controlVAE_170 = controlVAE(500, 3)
    controlVAE_200 = controlVAE(500, 3)
    controlVAE_170.load_state_dict(torch.load('controlVAE_170_model', map_location=torch.device('cpu')))
    controlVAE_200.load_state_dict(torch.load('controlVAE_200_model', map_location=torch.device('cpu')))

    train_data = load_celeba(50)
    images = iter(train_data).next()
    controlVAE_200_recon = reconstruct_images(images, controlVAE_200)
    controlVAE_170_recon = reconstruct_images(images, controlVAE_170)

    display_images(images, 'Original CelebA Input')
    display_images(controlVAE_200_recon, "controlVAE-200  Output")
    display_images(controlVAE_170_recon, ' controlVAE-170 Output')

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
