import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)
def corrplot(corrs):
    mask = np.triu(np.ones_like(corrs, dtype=np.bool))
    corrs[mask] = np.nan
    max_abs = np.nanmax(np.abs(corrs))
    plt.imshow(corrs, cmap=cmap, vmax=max_abs, vmin=-max_abs)