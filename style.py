import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

cmap_div = sns.diverging_palette(10, 220, as_cmap=True)
#
# plotting controls
#

c0 = '#1f77b4'
c1 = '#ff7f0e'
c2 = '#2ca02c'
c3 = '#d62728'
cs = [c0, c1, c2, c3]
cb = c0 #'#0084e3' #57, 138, 242)'
cr = '#d40035'


STANDARD_FIG_SIZE = (20,14)
VERY_SMALL_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 17
LARGE_SIZE = 20
VERY_LARGE_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rc('lines', lw=3)

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False