import time
import numpy as np
import torch
import torch.utils
import torch.distributions
import torch.nn.functional as F
import pickle

from Aux_functionc import plot_figure, kl_divergence, reconstruction_loss
from Image_Generation_MNIST.controlVAE import controlVAE
from PI_Contoller import pi_controller, PIDControl
from dataLoader import load_mnist

torch.manual_seed(0)


def train(vae, data_loader, desired_KL, vae_type='controlVAE', epochs=5):
    train_loss_avg = []
    train_kl_avg = []
    train_recon_avg = []
    train_beta_avg = []

    beta_max = 1
    beta_min = 0
    N = 1
    opt = torch.optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.99))

    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(data_loader):

            opt.zero_grad()
            x_recon, mu, log_var = vae(x)
            kl = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())).div(x.size(0))
            if vae_type == 'controlVAE':
                beta = pi_controller(desired_KL, kl.item(), beta_max, beta_min, N, Kp=0.01, Ki=0.0001)
            elif vae_type == 'VAE':
                beta = 1
            elif vae_type == 'betaVAE':
                beta = 100
            x_hat = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_hat, x, size_average=False).div(x.size(0))
            loss = recon_loss + beta * kl

            # # Paper implimintation
            # PID = PIDControl()
            # paper_kld = kl_divergence(mu, log_var)
            # beta_paper, _ = PID.pid(desired_KL, paper_kld.item(), 0.01, 0.001, 0)
            # paper_recon_loss = reconstruction_loss(x, x_recon)
            # loss_paper = paper_recon_loss + beta_paper * paper_kld

            loss.backward()
            opt.step()

            train_loss_avg.append(loss.item())
            train_kl_avg.append(kl.item())
            train_recon_avg.append(recon_loss.item())
            train_beta_avg.append(beta)

            if batch_idx % 20 == 0:
                print(
                    f'Batch {batch_idx} / {len(data_loader)} average loss error: {loss.item()}, average kld : {kl.item()}, average recon. error: {recon_loss.item()}, average beta: {beta}')
                # print(
                #     f'PAPER Batch {batch_idx} / {len(data_loader)} average loss error: {loss_paper.item()}, average kld : {paper_kld.item()}, average recon. error: {paper_recon_loss.item()}, average beta: {beta_paper}')
        print(
            f'Epoch {epoch + 1} / {epochs} average loss error: {np.mean(train_loss_avg)}, average kld : {np.mean(train_kl_avg)}, average recon. error: {np.mean(train_recon_avg)}, average beta: {np.mean(train_beta_avg)}')
    return vae, x_recon, train_loss_avg, train_recon_avg, train_kl_avg, train_beta_avg


def get_train_results(max_sim, batch_size, kl_set_point, model):
    vae = controlVAE(2, 1)
    train_data = load_mnist(batch_size)
    paper_max_steps = 1.2e6
    steps_per_epoch = len(train_data)
    epochs = int(paper_max_steps / steps_per_epoch)
    epochs = 50

    all_sim_elbo = []
    all_sim_recon_loss = []
    all_sim_kl = []
    all_sim_beta = []
    for numpy_seed in range(max_sim):
        np.random.seed(numpy_seed)
        torch_seed = np.random.randint(low=-2 ** 63,
                                       high=2 ** 63,
                                       dtype=np.int64)
        torch.manual_seed(torch_seed)
        vae, x_hat, elbo_list, recon_loss_list, kl_list, beta_list = \
            train(vae, train_data, kl_set_point, model, epochs)

        all_sim_elbo.append(elbo_list)
        all_sim_recon_loss.append(recon_loss_list)
        all_sim_kl.append(kl_list)
        all_sim_beta.append(beta_list)

    mean_elbo = np.mean(all_sim_elbo, axis=0)
    mean_recon = np.mean(all_sim_recon_loss, axis=0)
    mean_kl = np.mean(all_sim_kl, axis=0)
    mean_beta = np.mean(all_sim_beta, axis=0)

    filename = 'MNIST_output'
    outfile = open(filename, 'wb')
    pickle.dump({'all_sim_elbo': all_sim_elbo, 'all_sim_recon_loss': all_sim_recon_loss, 'all_sim_kl': all_sim_kl,
                 'all_sim_beta': all_sim_beta},
                outfile)
    outfile.close()
    torch.save(vae.state_dict(), 'vae_model')

    print('ContrloVAE ELBO {}+-{}'.format(np.mean(mean_elbo[-1]), np.std(mean_elbo[-1])))

    plot_figure([i for i in range(len(mean_elbo))], mean_elbo, 'Training Steps', 'Loss', f'Loss {model}')
    plot_figure([i for i in range(len(mean_recon))], mean_recon, 'Training Steps', 'Reconstruction Loss',
                f'ReconstructionLoss {model}')
    plot_figure([i for i in range(len(mean_kl))], mean_kl, 'Training Steps', 'KL Divergence', f'KL Divergence {model}')
    plot_figure([i for i in range(len(mean_beta))], mean_beta, 'Training Steps', 'Beta', f'Beta {model}')


def main():
    max_sim = 1  # how many simulations you need
    batch_size = 150
    kl_set_point = 170
    get_train_results(max_sim, batch_size, kl_set_point, 'VAE')
    # get_train_results(max_sim, batch_size, kl_set_point, 'betaVAE')
    # get_train_results(max_sim, batch_size, kl_set_point, 'controlVAE')


if __name__ == '__main__':
    main()
