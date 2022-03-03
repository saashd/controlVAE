import pickle

import numpy as np
import torch
import torch.utils
import torch.distributions
import torch.nn.functional as F

from Aux_functionc import plot_figure
from Disentangling.controlVAE import controlVAE
from PI_Contoller import pi_controller, PIDControl
from dataLoader import load_dsprites

torch.manual_seed(0)


def train(vae, data_loader, desired_KL, batch_size, epochs=5, period=5000, step_val=0.15, c=0.5):
    train_loss_avg = []
    train_kl_avg = []
    train_recon_avg = []
    train_beta_avg = []
    beta_max = 1
    beta_min = 0
    N = 1
    opt = torch.optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.99))
    iters = 0
    for epoch in range(epochs):
        train_loss_avg.append(0)
        train_kl_avg.append(0)
        train_recon_avg.append(0)
        train_beta_avg.append(0)

        for batch_idx, x in enumerate(data_loader):
            iters += 1
            opt.zero_grad()
            x_recon, mu, log_var = vae(x)

            # Alex implimintation
            if iters % period == 0:
                c += step_val
            if c > desired_KL:
                c = desired_KL
            print('set_point: {}'.format(c))
            kl = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())).div(batch_size)
            beta = pi_controller(c, kl.item(), beta_max, beta_min, N, Kp=0.01, Ki=0.001)
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
            loss = recon_loss + beta * kl

            # # Paper implimintation
            # PID = PIDControl()
            # paper_kld = kl_divergence(mu, log_var)
            # beta_paper, _ = PID.pid(c, paper_kld.item(), 0.01, 0.001, 0)
            # paper_recon_loss = reconstruction_loss(x, x_recon)
            # loss_paper = paper_recon_loss + beta_paper * paper_kld
            #
            # elbo_list.append(loss.item())
            # recon_loss_list.append(recon_loss.item())
            # kl_list.append(kl.item())
            # beta_list.append(beta)

            loss.backward()
            opt.step()

            train_loss_avg[-1] += loss.item()
            train_kl_avg[-1] += kl.item()
            train_recon_avg[-1] += recon_loss.item()
            train_beta_avg[-1] += beta

            if iters % 20 == 0:
                print(
                    f'Batch {batch_idx} / {len(data_loader)} average loss error: {loss.item()}, average kld : {kl.item()}, average recon. error: {recon_loss.item()}, average beta: {beta}')
        print(
            f'Epoch {epoch + 1} / {epochs} average loss error: {np.mean(train_loss_avg)}, average kld : {np.mean(train_kl_avg)}, average recon. error: {np.mean(train_recon_avg)}, average beta: {np.mean(train_beta_avg)}')
    return vae, x_recon, train_loss_avg, train_recon_avg, train_kl_avg, train_beta_avg


def main():
    max_sim = 1  # how many simulations you need

    vae = controlVAE(10, 1)
    params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('# of vae trainable params: {}'.format(params))

    batch_size = 64
    train_data = load_dsprites(batch_size)
    kl_set_point = 16
    c = 0.5
    period = 5000  # K
    step_val = 0.15  # alpha or s

    paper_max_steps = 1.2e6
    steps_per_epoch = len(train_data)
    epochs = int(paper_max_steps / steps_per_epoch)

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
            train(vae, train_data, kl_set_point, batch_size, epochs, period, step_val, c)

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
    torch.save(vae.state_dict(), filename)

    print('ContrloVAE ELBO {}+-{}'.format(np.mean(mean_elbo[-1]), np.std(mean_elbo[-1])))

    plot_figure([i for i in range(len(mean_elbo))], mean_elbo, 'Training Steps', 'ELBO', 'ELBO')
    plot_figure([i for i in range(len(mean_recon))], mean_recon, 'Training Steps', 'Reconstruction Loss',
                'ReconstructionLoss')
    plot_figure([i for i in range(len(mean_kl))], mean_kl, 'Training Steps', 'KL Divergence', 'KL Divergence')
    plot_figure([i for i in range(len(mean_beta))], mean_beta, 'Training Steps', 'Beta', 'Beta')


if __name__ == '__main__':
    main()
