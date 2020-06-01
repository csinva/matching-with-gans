import matplotlib.pyplot as plt
import seaborn as sns
cmap_div = sns.diverging_palette(10, 220, as_cmap=True)
#
# plotting controls
#
STANDARD_FIG_SIZE = (20,14)
VERY_SMALL_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 20
LARGE_SIZE = 26
VERY_LARGE_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=VERY_LARGE_SIZE)  # fontsize of the figure title
plt.rc('lines', lw=3)