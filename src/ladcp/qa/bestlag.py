"""Port of the LDEO_IX ``bestlag.m`` — robust integer lag between two ~1 Hz series.

Finds the shift of ``a2`` relative to ``a1`` that best aligns them, using the median of
the absolute gradient difference (robust to spikes and amplitude offsets), refined by a
local parabola fit, then reports the correlation at that lag after iterative spike
removal. Used to synchronise the CTD vertical-velocity series to the LADCP one.
"""

from __future__ import annotations

import numpy as np

from .stats import medianan


def bestlag(a1: np.ndarray, a2: np.ndarray, nlag: int | None = None) -> tuple[int, float]:
    """Return (lag, correlation). ``lag`` shifts ``a2`` onto ``a1`` (a2[i+lag] ~ a1[i])."""
    a1 = np.asarray(a1, float).ravel()
    a2 = np.asarray(a2, float).ravel()
    l1, l2 = a1.size, a2.size
    lmin = min(l1, l2)
    if nlag is None:
        nlag = (l1 + l2) // 8
    ilag = np.arange(-nlag, nlag + 1)

    lag1, lag2 = ilag.min(), ilag.max()
    ic = np.arange(max(1, -lag1) + 1, lmin - max(0, lag2) - 1) - 1  # 0-based target idx
    if ic.size < 10:
        return 0, 1.0

    c1 = np.tile(a1[ic][:, None], (1, ilag.size))          # [npt, nlag]
    c2 = np.empty_like(c1)
    for i, lg in enumerate(ilag):
        c2[:, i] = a2[ic + lg]

    good = np.isfinite(c1.sum(1) + c2.sum(1))
    if good.sum() < 10:
        return 0, 1.0
    c1g, c2g = c1[good], c2[good]

    # median of |gradient(c2) - gradient(c1)| per lag column
    g1 = np.gradient(c1g, axis=0)
    g2 = np.gradient(c2g, axis=0)
    diff = np.abs(g2 - g1)
    na = int(np.ceil(good.sum() / 6))
    cm = np.array([medianan(diff[:, i], na) for i in range(ilag.size)])
    im1 = int(np.argmin(cm))

    # parabola refinement around the minimum
    il = ilag.size // 4
    ifit = np.arange(-il, il + 1) + im1
    ifit = ifit[(ifit >= 0) & (ifit < ilag.size)]
    im = im1
    if ifit.size > 10:
        c = np.polyfit(ilag[ifit], cm[ifit], 2)
        cmb = cm + np.mean(cm) * 2
        cmf = np.full_like(cm, np.nan)
        cmf[ifit] = np.polyval(c, ilag[ifit])
        cmb[ifit] = cmf[ifit] / 6 + 5 / 6 * cm[ifit]
        im = int(np.argmin(cmb))

    # correlation at best lag with iterative spike removal
    c3 = np.column_stack([c1g[:, 0], c2g[:, im]])
    for i in range(1, 5):
        c3d = np.diff(c3, axis=0)
        keep = np.abs(c3d) <= (2 * (1 + i / 8) * np.std(c3d, axis=0))
        keep_rows = np.concatenate([[True], keep.all(axis=1)])
        if (keep_rows.sum()) < 0.8 * good.sum():
            break
        c3 = c3[keep_rows]
    co = float(np.corrcoef(c3, rowvar=False)[0, 1]) if c3.shape[0] > 2 else 0.0
    return int(ilag[im]), co
