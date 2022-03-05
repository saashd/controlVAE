import pickle
import torch
import torch.utils
import torch.distributions
import torch.nn.functional as F

from Image_Generation_celebA.controlVAE import controlVAE
from PI_Contoller import PI_Controller
from dataLoader import load_celeba

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(vae, data_loader, desired_KL, vae_type, epochs=5):
    train_kl = []
    train_recon = []

    opt = torch.optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.99))
    x_recon = None
    PI = PI_Controller()
    for epoch in range(epochs):
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)
            x_recon, mu, log_var = vae(x)

            kl = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())).div(x.size(0))
            beta = 1
            if vae_type == 'controlVAE':
                beta = PI.pi(desired_KL, kl.item(), 1, 0, 1, Kp=0.01, Ki=0.0001)
            elif vae_type == 'betaVAE':
                beta = 100
            x_hat = torch.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_hat, x, reduction='sum').div(x.size(0))
            loss = recon_loss + beta * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_kl.append(kl.item())
            train_recon.append(recon_loss.item())

        if epoch % 50 == 0 and epoch != 0:
            filename = f'celebA_output_{vae_type}_{desired_KL}'
            outfile = open(filename, 'wb')
            pickle.dump(
                {'train_recon': train_recon,
                 'train_kl': train_kl},
                outfile)
            outfile.close()
            torch.save(vae.state_dict(), f'{vae_type}_model')
        print(
            f'Epoch {epoch + 1} / {epochs} '
            f'  kld : {kl.item()},'
            f'  recon. error: {recon_loss.item()}')
    return vae, x_recon, train_recon, train_kl


def get_train_results(batch_size, kl_set_point, model):
    vae = controlVAE(500, 3).to(device)
    # vae.load_state_dict(torch.load('vae_model'))
    train_data = load_celeba(batch_size)
    epochs = 5

    vae, x_hat, recon_loss_list, kl_list = \
        train(vae=vae, data_loader=train_data, desired_KL=kl_set_point, vae_type=model, epochs=epochs)

    filename = f'celebA_output_{model}_{kl_set_point}'
    outfile = open(filename, 'wb')
    data = {'recon_loss_list': recon_loss_list, 'kl_list': kl_list}
    print(data)
    pickle.dump(data, outfile)
    outfile.close()
    torch.save(vae.state_dict(), f'{model}_{kl_set_point}_model')


def main():
    batch_size = 100
    get_train_results(batch_size, 100, 'betaVAE')
    get_train_results(batch_size, 0, 'VAE')
    get_train_results(batch_size, 200, f'controlVAE')
    get_train_results(batch_size, 180, f'controlVAE')
    get_train_results(batch_size, 170, f'controlVAE')


if __name__ == '__main__':
    main()
