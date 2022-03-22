import pickle
import torch
import torch.utils
import torch.distributions
import torch.nn.functional as F

from Disentangling.PI_Contoller import PI_Controller
from Disentangling.controlVAE import controlVAE
from dataLoader import load_dSprites

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


def train(vae, data_loader, desired_KL, vae_type, epochs=5, period=5000, step_val=0.15, c=0.5):
    train_loss = []
    train_kl = []
    train_recon = []
    train_beta = []
    opt = torch.optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.99))
    x_recon = None
    PI = PI_Controller()
    iters = 0
    for epoch in range(epochs):

        for batch_idx, (x, y) in enumerate(data_loader):
            iters += 1
            x = x.to(device)
            x_recon, mu, log_var = vae(x)

            if iters % period == 0:
                c += step_val
            if c > desired_KL:
                c = desired_KL

            kl = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())).div(x.size(0))

            beta = 1
            if vae_type == 'controlVAE':
                beta = PI.pi(c, kl.item(), 1, 1, 0.01, 0.001)
            elif vae_type == 'betaVAE':
                beta = 100
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(x.size(0))
            loss = recon_loss + beta * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())
            train_kl.append(kl.item())
            train_recon.append(recon_loss.item())
            train_beta.append(beta)

        if epoch % 100 == 0 and epoch != 0:
            filename = f'dSprites_output_{vae_type}'
            outfile = open(filename, 'wb')
            pickle.dump(
                {'train_loss': train_loss,
                 'train_recon': train_recon,
                 'train_kl': train_kl,
                 'train_beta': train_beta},
                outfile)
            outfile.close()
            torch.save(vae.state_dict(), f'{vae_type}_model')
        print(
            f'Epoch {epoch + 1} / {epochs}  loss error: {loss.item()},'
            f'kld : {kl.item()},'
            f'recon. error: {recon_loss.item()},'
            f'beta: {beta}')
    return vae, x_recon, train_loss, train_recon, train_kl, train_beta


def get_train_results(batch_size, kl_set_point, model):
    c = 0.5
    period = 1000  # K
    step_val = 0.15  # alpha or s
    vae = controlVAE(10, 1).to(device)
    # vae.load_state_dict(torch.load('controlVAE_16_model'))
    train_data = load_dSprites(batch_size=batch_size)
    epochs = 600

    vae, x_hat, loss_list, recon_loss_list, kl_list, beta_list = \
        train(vae=vae, data_loader=train_data, desired_KL=kl_set_point, vae_type=model, epochs=epochs, period=period,
              step_val=step_val, c=c)

    filename = f'dSprites_output_{model}_{kl_set_point}'
    outfile = open(filename, 'wb')
    data = {'loss_list': loss_list,
            'recon_loss_list': recon_loss_list,
            'kl_list': kl_list,
            'beta_list': beta_list}
    print(data)
    pickle.dump(data, outfile)
    outfile.close()
    torch.save(vae.state_dict(), f'{model}_{kl_set_point}_model')


def main():
    batch_size = 64
    get_train_results(batch_size, 16, f'controlVAE')
    # get_train_results( batch_size, 18, f'controlVAE')
    # get_train_results(batch_size, 100, 'betaVAE')


if __name__ == '__main__':
    main()
