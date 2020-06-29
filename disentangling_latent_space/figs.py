from os.path import join as oj

import matplotlib.pyplot as plt
import pandas as pd

import style

PROCESSED_DIR = 'processed'
RESULTS_DIR = 'results'
DIRECTIONS_DIR = '../data/annotation-dataset-stylegan2/linear_models/new'  # path to many datasets, includes on directory before the below dirs
GENERATING_LATENTS_DIR = '../data/annotation-dataset-stylegan2/data'


def plot_mse_vs_corrs():
    fnames = {
        '06_df_loss_tradeoff_linear.pkl': 'Linear',
        '06_df_loss_tradeoff_nonlinear.pkl': 'MLP',
        #         '07_df_loss_tradeoff_nonlinear_INN.pkl': 'Nonlinear-INN',
        #         '07_df_loss_tradeoff_nonlinear_INN_wide.pkl': 'Nonlinear-INN-Wide',
        #         '07_df_loss_tradeoff_nonlinear_INN_2lay.pkl': 'Nonlinear-INN-2Lay',
        #         '07_df_loss_tradeoff_nonlinear_INN_3lay.pkl': 'Nonlinear-INN-3Lay',
        #         '07_df_loss_tradeoff_nonlinear_INN_8lay.pkl': 'Nonlinear-INN-8Lay',
        #         '07_df_loss_tradeoff_nonlinear_INN_affine.pkl': 'Real-NVP',
        #         '07_df_loss_tradeoff_nonlinear_INN_RNVP_1lay.pkl': 'Tanh-Reverse-Order',
        #         '07_tanh_3lay.pkl': 'Tanh-3lay',
        #         '07_tanh_.pkl': 'Tanh-3lay-2',
        #         '07_df_loss_tradeoff_nonlinear_INN_RNVP_8lay.pkl': '8Lay',
        #         '07_relu_retrain_3lay.pkl': 'Relu-retrain-3-lay',
        #         '07_relu_retrain_3lay_noise.pkl': 'Relu-retrain-3-lay-noise',
        '07_relu_retrain_3lay_noise_big.pkl': 'INN',
        #         '07_relu_retrain_3lay_noise_huge.pkl': 'Relu-retrain-3-lay-noise-huge',
        #         '07_relu_retrain_3lay_noise_wide.pkl': 'Relu-retrain-3-lay-noise-wide',
        #         '07_relu_retrain_3lay_noise_wide_deep.pkl': 'Relu-retrain-3-lay-noise-wide_deep',

    }

    '''
    R, C = 1, 2
    plt.figure(figsize=(12, 5), dpi=500)
    for k, v in fnames.items():
        df = pd.read_pickle(oj(PROCESSED_DIR, k))
        df = df.iloc[1:]
        plt.subplot(R, C, 1)
        plt.plot(df['mse'], df['indep_corr'], 'o-', label=v)
        plt.xlabel('Mean-squared error')
        plt.ylabel('corr. between attributes')
        plt.title('Training')
        plt.xlim((-0.1, 1))

        plt.subplot(R, C, 2)
        plt.plot(df['mse_test'], df['indep_corr_test'], 'o-', label=v)
        plt.xlabel('Mean-squared error')
        plt.title('Testing')
        plt.xlim((0.1, 0.7))
    '''

    R, C = 1, 1
    plt.figure(figsize=(12, 5), dpi=500)
    for i, (k, v) in enumerate(fnames.items()):
        df = pd.read_pickle(oj(PROCESSED_DIR, k))
        df = df.iloc[1:]
        #         plt.plot(df['spearman'], df['indep_corr'], 'o:', label=v + ' (Test)', color=style.cs[i])
        #         plt.plot(df['mse'], df['indep_corr'], 'o:', label=v + ' (Train)', color=style.cs[i], alpha=0.5)
        plt.xlabel('Mean-squared error (Test)')
        plt.ylabel('Mean absolute\ninter-attribute correlation (Test)')

        #         plt.plot(df['mse_test'], df['indep_corr_test'], 'o-', label=v + ' (Test)', color=style.cs[i])
        plt.plot(df['mse_test'], df['indep_corr_test'], 'o-', label=v, color=style.cs[i])
        #         plt.plot(df['spearman_test'], df['indep_corr_test'], 'o-', label=v + ' (Test)', color=style.cs[i])
        plt.xlabel('Mean squared-error')
    #     plt.xlim((0, 0.7))
    plt.xlim((0.2, 0.7))

    # grid
    ax = plt.subplot(R, C, 1)
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.3)
    #     plt.grid()

    # previous 
    MSE_ORIG = 0.3671995300361915
    CORR_ORIG = 0.5514424823248604
    plt.plot(MSE_ORIG, CORR_ORIG, 'x', color='black', ms=10)
    ax.annotate('Separate linear models', (MSE_ORIG + 0.02, CORR_ORIG), color='black')

    MSE_ORTH = 0.51532
    CORR_ORTH = 0.340195386
    plt.plot(MSE_ORTH, CORR_ORTH, 'x', color='black', ms=10)
    ax.annotate('Orthogonalized linear models', (MSE_ORTH + 0.02, CORR_ORTH), color='black')

    plt.subplot(R, C, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig(oj(RESULTS_DIR, 'fig_attr_mse.pdf'))
    plt.show()