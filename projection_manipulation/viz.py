import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')

import data
from config import *
from style import *
import util

df = data.load_all_labs()
df = df.set_index('fname_id')

# get fnames
fname_nps = [f for f in sorted(os.listdir(DIR_GEN)) if 'npy' in f] # these start at 00001
fname_ids = np.array([f[:-4] for f in fname_nps])
idxs_calculated = np.array([int(x) - 1 for x in fname_ids]) # this starts at 0

# trim df to only have the relevant ids
df = df.loc[fname_ids]

def visualize_projection_statistics(df, d, vals):
    '''Visualize statistics as a function of regularization
    '''
    arrow_args = {
        'arrowprops': {
            'facecolor': 'black',
            'width': 1,
            'headwidth': 8,
            'shrink': 0.05
        },
        'xycoords': 'data',
        'textcoords': 'data',
        'horizontalalignment': 'left',
        'verticalalignment': 'center',
        'fontsize': 16
    }
    ykey = 'mean_abs_corr'
    plt.figure(dpi=300, figsize=(13, 4))
    plt.subplot(121)
    means = df.groupby('reg_param').mean().reset_index()
    ax = plt.gca()
    plt.plot(means['perceptual_loss'], means[ykey], 'o-')

    # annotate 0
    r0 = means.iloc[0]
    (x, y) = r0['perceptual_loss'], r0[ykey]
    # plt.text(x0, y0, 'No regularization')
    ax.annotate('Unregularized',
                xy=(x, y),
                xytext=(x + 0.005, y), 
                **arrow_args)

    # annotate 1
    r1 = means.iloc[2]
    (x, y) = r1['perceptual_loss'], r1[ykey]
    plt.plot(r1['perceptual_loss'], r1[ykey], marker='*', ms=13)
    # plt.text(x1, y1, 'Regularization = $10^{-1}$')
    ax.annotate('Good amount of regularization', # = $10^{-1}$',
                xy=(x, y),
                xytext=(x + 5e-3, y - 8e-2), 
                **arrow_args)

    # annotate 2
    r1 = means.iloc[-1]
    (x, y) = r1['perceptual_loss'], r1[ykey]
    # plt.plot(r1['perceptual_loss'], r1['mean_abs_corr'], marker='*', ms=13)
    # plt.text(x, y, 'Unexpanded latent space')
    arrow_args['horizontalalignment'] = 'right'
    ax.annotate('Restricted',
                xy=(x, y),
                xytext=(x - 1e-3, y - 15e-2), 
                **arrow_args,
               )

    plt.ylabel('Closeness to restricted\nstyle space')
    plt.xlabel('Perceptual loss')
    # util.savefig('projection_perceptual_loss')


    # right subplot
    plt.subplot(122)
    arrow_args = {
        'arrowprops': {
            'facecolor': 'black',
            'width': 1,
            'headwidth': 8,
            'shrink': 0.05
        },
        'xycoords': 'data',
        'textcoords': 'data',
        'horizontalalignment': 'left',
        'verticalalignment': 'center',
    }
    ykey = 'mean_abs_corr'
    plt.plot(d[vals].mean(), means[ykey], 'o-')
    plt.plot(d[vals].mean()[2], means[ykey][2], marker='*', ms=13)

    plt.xlabel('Facial ID distance between original and\nprojected image encoding')

def visualize_projection_statistics_overlapping(df, d, vals):
    '''Visualize statistics as a function of regularization
    '''
    ykey = 'mean_abs_corr'
    plt.figure(dpi=300, figsize=(9, 5))
#     plt.subplot(121)
    means = df.groupby('reg_param').mean().reset_index()

    x1 = means['perceptual_loss'] - means['perceptual_loss'].min()
    plt.plot(x1, np.arange(5), 'o-', color=c0)
    plt.text(x1.max() + 0, 4.25, 'Perceptual distance (VGG)', color=c0, fontsize=18)
    
    plt.yticks(np.arange(5), labels=['Unregularized: $\lambda = 0$',
                                     '$\lambda = 0.001$', 
                                     '$\mathbf{Regularized: \lambda = 0.1}$',
                                     '$\lambda = 1$',
                                     'Restricted: $\lambda = \infty$'])
    plt.text(x1[2] - 0.006, 1.8, '0.34', color=c0, fontsize=14)
    plt.text(x1[0] - 0.006, -0.08, '0.31', color=c0, fontsize=14)
    
    x2 = (d[vals].mean() - d[vals].mean().min()) / 11 + 0.003
    plt.plot(x2, np.arange(5), 'o-', color=c1)    
    plt.text(x2.max() + 0.001, 3.1, 'Face-rec distance\n(dlib)', color=c1, fontsize=18)
    plt.text(x2[2] + 0.0015, 1.8, '0.731', color=c1, fontsize=14)
    plt.text(x2[0] + 0.0015, -0.08, '0.733', color=c1, fontsize=14)
    
    plt.xlabel('Distance between original and reconstructed images\n(lower is better)')
    plt.xticks([])
    plt.xlim((-0.007, 0.04))
    plt.ylim((-0.25, 4.8))
#     print(means['perceptual_loss'], d[vals].mean())

    
def hist_subgroups(means, labs, labs_list, BINS=4):
    for lab in labs_list:
        plt.hist(means[labs[lab]], BINS, alpha=0.5, label=lab)
    plt.ylabel('Count')
    plt.xlabel('Fraction of images said to be "Same"')
    plt.legend()
    plt.show()

def boxplot_subgroups(vals, labs, labs_list, confs='sem', ret=True, width=7):
    '''
    vals: array_like
        what to plot
    labs: dataframe / dict
        contains indexes over which to extract the vals
    labs_list: array_like
        labels for the means
    confs: array_like
        confidence intervals
    '''
    plt.figure(dpi=300, figsize=(width, 3))
    lists = [vals[labs[lab]] for lab in labs_list]
    ys = np.arange(1, len(labs_list) + 1)
    if confs is None:
        plt.boxplot(lists, vert=False, showmeans=True)
    else:
        means = [np.mean(l) for l in lists]
        if confs == 'sem':
            sems = [1.96 * np.std(l) / np.sqrt(len(l)) for l in lists]
        elif confs == 'wilson':
            wilsons = [util.wilson(l) for l in lists]
            sems = np.abs(np.array([[w[0], w[1]] for w in wilsons]).transpose())
        plt.errorbar(means,
                     ys, xerr=sems,
                     linestyle='None', marker='o', ms=8)
        # plt.boxplot(lists, vert=False, showmeans=True)
    plt.yticks(ys, [x.capitalize() for x in labs_list])
    plt.xlabel('Fraction of pairs labelled as "Same"')
    plt.ylabel('True conditions')
    plt.xlim((0.3, 1.05))
    if ret:
        return np.array(means), np.array(sems).transpose()
    
def annotators_num_plot(annotations):
    n_annotations = sorted(annotations.annotations, reverse=True)
    plt.figure(dpi=100)
    plt.grid()
    plt.plot(range(1, 1 + annotations.N_ANNOTATORS), n_annotations)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Number of annotators')
    plt.ylabel('Number of annotations')
    plt.title('Work of individual annotators')
    plt.show()    
    
    
def print_table(means_list, sems_list, index=None):
    '''
    Params
    ------
    means_list: (n,)
    sems_list: (n, 2)
    '''
    index = np.array(index)
    
    
    # print table with sem CI
    # strs = np.array([f'{100 * means_tab[i]:0.1f} $\pm$ {100 * sems_tab[i]:0.2f}'
    #                  for i in range(len(means_tab))])
    
    # print table with wilson CI
    strs = np.array([f'{100 * means_list[i]:0.1f} ' + \
                     f'({100 * (means_list[i] - sems_list[i, 0]):0.2f}, {100 * (means_list[i] + sems_list[i, 1]):0.2f})'
                     for i in range(len(means_list))])
    
    
    # put into df
    d = pd.DataFrame(strs.reshape(-1, 2)) #, columns) #, columns=l_non_dup)
    
#     print(d.size, columns.size)
    if index.size == d.shape[0] * 2:
        index = index[::2]
    index = [i.capitalize().replace(' (fake)', '').replace(' (real)', '') for i in index]
    
#     print(d, columns)
    # d.columns = columns

    d.index = index
    d.columns = ['Fake pairs', 'Real pairs']
    
    s = d.to_latex(index=True).replace('textbackslash pm', 'pm').replace('\$', '$')
    s = s.replace('bottomrule', 'bottomrule\\\\') # add space at end
    s = s.replace('Knows well', '\n\midrule\nKnows well') # add space before knows well
    print(s)