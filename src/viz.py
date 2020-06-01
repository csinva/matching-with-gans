import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import style
# Generate a custom diverging colormap

def corrplot(corrs):
    mask = np.triu(np.ones_like(corrs, dtype=np.bool))
    corrs[mask] = np.nan
    max_abs = np.nanmax(np.abs(corrs))
    plt.imshow(corrs, cmap=style.cmap_div, vmax=max_abs, vmin=-max_abs)