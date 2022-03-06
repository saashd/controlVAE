from Aux_functionc import plot_figure


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


if __name__ == '__main__':
    main()
