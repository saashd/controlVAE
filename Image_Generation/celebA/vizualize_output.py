from Aux_functions import plot_figure


def main():
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
