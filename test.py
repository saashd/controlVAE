import numpy as np
import torch
import torch.utils
import torch.distributions
import torch.nn.functional as F
import torchvision
from matplotlib import ticker

from controlVAE import controlVAE
from PI_Contoller import pi_controller
from dataLoader import load_celeba, load_cifar10
import matplotlib.pyplot as plt

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(0)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose((npimg * 255).astype('uint8'), (1, 2, 0)))
    plt.show()


def train(vae, data_loader, desired_KL, batch_size, epochs=5):
    beta_max = 1
    beta_min = 0
    N = 1
    opt = torch.optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.99))
    elbo_list = []
    rec_loss_list = []
    kl_list = []
    for epoch in range(epochs):
        print(epoch)
        for x in data_loader:
            # imshow(torchvision.utils.make_grid(x))
            opt.zero_grad()
            x_recon, mu, log_var = vae(x)
            # imshow(torchvision.utils.make_grid(x_recon))
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            beta = pi_controller(desired_KL, kl, beta_max, beta_min, N, Kp=0.01, Ki=0.0001)
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
            loss = recon_loss + beta * kl
            elbo_list.append(loss.item())
            rec_loss_list.append(recon_loss.item())
            kl_list.append(kl.item())
            loss.backward()
            opt.step()
    print('ELBO: {}'.format(elbo_list))
    print('REC loss: {}'.format(rec_loss_list))
    print('KL_d: {}'.format(kl_list))
    return vae, x_recon, elbo_list, rec_loss_list, kl_list


def plot_figure(x, y, x_title, y_title, fig_title):
    fig, ax = plt.subplots()
    plt.scatter(x, y)
    plt.tick_params(labelsize=15)
    plt.xlabel(x_title, fontsize=15)
    plt.ylabel(y_title, fontsize=15)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x / 100)) + 'K'))
    plt.legend(loc='best', prop={'size': 11.5})
    plt.grid()
    plt.tight_layout()
    fig.savefig(fig_title, bbox_inches='tight', dpi=600)
    plt.show()


def main():
    dataset = input("Enter dataset for a  train (celebA,CIFAR10) :")

    max_sim = 5  # how many simulations you need
    epochs = 4
    if dataset == 'celebA':
        batch_size = 128
        vae = controlVAE(500, 3)
        train_data, test_data = load_celeba(batch_size)
        kl_set_point = 155
    elif dataset == 'CIFAR10':
        batch_size = 100
        vae = controlVAE(200, 3)
        train_data, test_data = load_cifar10(batch_size)
        kl_set_point = 145
    else:
        return

    all_sim_elbo = []
    all_sim_rec_loss = []
    all_sim_kl = []
    for numpy_seed in range(max_sim):
        np.random.seed(numpy_seed)
        torch_seed = np.random.randint(low=-2 ** 63,
                                       high=2 ** 63,
                                       dtype=np.int64)
        torch.manual_seed(torch_seed)
        vae, x_hat, elbo_list, rec_loss_list, kl_list = train(vae, train_data, kl_set_point, batch_size,
                                                              epochs)

        all_sim_elbo.append(elbo_list)
        all_sim_rec_loss.append(rec_loss_list)
        all_sim_kl.append(kl_list)

    mean_elbo = np.mean(all_sim_elbo, axis=0)
    mean_rec = np.mean(all_sim_rec_loss, axis=0)
    mean_kl = np.mean(all_sim_kl, axis=0)

    print('ContrloVAE ELBO {}+-{}'.format(np.mean(mean_elbo[-1]), np.std(mean_elbo[-1])))

    plot_figure([i for i in range(len(mean_elbo))], mean_elbo, 'Training Steps', 'ELBO', 'ELBO')
    plot_figure([i for i in range(len(mean_rec))], mean_rec, 'Training Steps', 'Reconstruction Loss',
                'ReconstructionLoss')
    plot_figure([i for i in range(len(mean_kl))], mean_kl, 'Training Steps', 'KL Divergence', 'KL Divergence')


if __name__ == '__main__':
    main()
