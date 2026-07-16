"""Full-bleed raw-matrix heatmap for the Studio Edit view (no axes/margins).

The client maps pixels to (ensemble, bin) purely fractionally, so the figure is a
single axes-less ``imshow``. Auto-screened cells are dimmed so the user sees what the
pipeline already rejects.
"""
from __future__ import annotations

import numpy as np


def edit_heatmap_figure(mat, auto=None, view: str = "errvel"):
    """``mat`` (bins x ensembles) as a full-bleed figure; ``auto`` dims screened cells."""
    import matplotlib.pyplot as plt

    fin = mat[np.isfinite(mat)]
    vmin, vmax = (np.percentile(fin, [2, 98]) if fin.size else (0.0, 1.0))
    fig = plt.figure(figsize=(10.0, 4.2), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(mat, aspect="auto", origin="upper", interpolation="nearest",
              cmap="magma" if view == "errvel" else "viridis",
              vmin=vmin, vmax=vmax)
    if auto is not None and auto.any():          # dim what auto-screening already removed
        dim = np.zeros((*auto.shape, 4))
        dim[auto] = (0.55, 0.58, 0.62, 0.75)
        ax.imshow(dim, aspect="auto", origin="upper", interpolation="nearest")
    return fig
