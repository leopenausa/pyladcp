"""Own bottom-track from echo amplitude (getbtrack.m / localmax2.m / targ).

The RDI hardware bottom-track range is noisy in deep water (false / sidelobe locks
that imply the instrument *below* the seafloor), so LDEO_IX builds its **own** bottom
distance from the down-looker echo: convert echo amplitude to range-compensated target
strength (``targ``), find the parabolic peak of that profile per ensemble (``localmax2``),
and accept it as the bottom distance where the peak stands clearly above the bin-1 level.
For ``btrk_mode==2`` (the GO-SHIP golden setting) this own distance replaces the RDI one.

This is the cleaned ``hbot`` consumed by :func:`ladcp.ix.depth.detect_bottom_and_flag`.
Faithful to ``getbtrack.m`` (mode 2/3 path); the bottom-track *velocity* assembly and the
W-/outlier checks on it are out of scope here (they feed the inverse, not depth).
"""

from __future__ import annotations

import numpy as np

from .ingest import LADCPData, _quiet_nan


def target_strength(
    echo_db: np.ndarray,
    dist: np.ndarray,
    at: float,
    *,
    bin_len: float | None = None,
    source_level: float = 100.0,
    aperture_deg: float = 2.0,
) -> np.ndarray:
    """Range-compensated target strength of a volume scatterer (``targ`` in getbtrack.m).

    ``echo_db`` is echo amplitude in dB ``[nbin, nens]``; ``dist`` the bin distances [m]
    ``[nbin]``; ``at`` the absorption coefficient [dB/m] (0.039 at 150 kHz, 0.06 at
    300 kHz). Returns target strength [dB] with the same shape as ``echo_db``.
    """
    ea = np.asarray(echo_db, float)
    nbin, nens = ea.shape
    d = np.asarray(dist, float).reshape(-1, 1) * np.ones((1, nens))
    if bin_len is None:
        bin_len = float(np.median(np.abs(np.diff(d[:, 0]))))
    al = aperture_deg * np.pi / 180.0
    r1 = np.tan(al) * (d - bin_len / 2)
    r2 = np.tan(al) * (d + bin_len / 2)
    vol = np.pi * bin_len / 3.0 * (r1**2 + r2**2 + r1 * r2)   # ensonified volume
    tl = 20.0 * np.log10(d) + at * d                          # transmission loss
    return ea - source_level + 2.0 * tl - 10.0 * np.log10(vol)


def localmax2(dist: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic sub-bin peak of each column (faithful ``localmax2.m``).

    ``dist`` ``[nbin]`` (shared distance axis), ``y`` ``[nbin, nens]``. Returns
    ``(x_peak, y_peak)`` ``[nens]``: the interpolated distance/value of the maximum, or
    NaN where the discrete max is at an endpoint or the fitted parabola opens upward.
    """
    nbin, nens = y.shape
    with _quiet_nan():
        imax = np.nanargmax(np.where(np.isfinite(y), y, -np.inf), axis=0)
    xout = np.full(nens, np.nan)
    yout = np.full(nens, np.nan)
    iok = np.where((imax > 0) & (imax < nbin - 1))[0]
    if iok.size == 0:
        return xout, yout
    k = imax[iok]
    x1, x2, x3 = dist[k - 1], dist[k], dist[k + 1]
    y1, y2, y3 = y[k - 1, iok], y[k, iok], y[k + 1, iok]
    a = ((x3 * y2 + x1 * y3 - x1 * y2 - y3 * x2 - y1 * x3 + y1 * x2)
         / (x3 * x2**2 - x1 * x2**2 + x1 * x3**2 - x1**2 * x3 + x1**2 * x2 - x3**2 * x2))
    b = (-(-x2**2 * y3 + x2**2 * y1 - y2 * x1**2 + y3 * x1**2 - x3**2 * y1 + y2 * x3**2)
         / ((-x3 + x2) * (x2 * x3 - x2 * x1 + x1**2 - x3 * x1)))
    c = ((x2**2 * y1 * x3 - x2**2 * x1 * y3 - x3**2 * y1 * x2
          + y3 * x1**2 * x2 + x3**2 * x1 * y2 - y2 * x1**2 * x3)
         / ((-x3 + x2) * (x2 * x3 - x2 * x1 + x1**2 - x3 * x1)))
    with _quiet_nan():
        opens_down = a < 0
        xn = np.where(opens_down, -b / a / 2.0, np.nan)
        yn = xn**2 * a + xn * b + c
    xout[iok] = xn
    yout[iok] = yn
    return xout, yout


def own_bottom_track(
    d: LADCPData,
    *,
    echo_scale: float = 0.45,
    btrk_ts: float = 10.0,
    btrk_range: tuple[float, float] = (300.0, 50.0),
    btrk_below: float = 0.5,
) -> np.ndarray:
    """Own bottom distance ``hbot`` [m] from down-looker echo (getbtrack mode 2/3).

    Steps (``getbtrack.m``): echo amplitude (counts ``->`` dB via ``echo_scale``) ``->``
    target strength ``->`` per-ensemble parabolic peak distance ``zmead``; accept where the
    peak rises ``> btrk_ts`` dB above bin 1, the distance is inside ``btrk_range``, and the
    velocity bin sits inside the profile. Returns ``hbot`` ``[nens]`` (NaN where no bottom).

    The down-head frequency selects the absorption coefficient (0.06 dB/m at ``>=`` 300 kHz,
    else 0.039); taken from ``d.meta['freq_dn_khz']``.
    """
    freq = float(d.meta.get("freq_dn_khz", 150.0))
    at = 0.06 if freq >= 300 else 0.039

    izd, zd = d.izd, d.zd
    nbin = izd.size
    echo_db = d.ts[izd, :] * echo_scale          # [nbin_d, nens] dB
    tg = target_strength(echo_db, zd, at)
    zmead, mead = localmax2(zd, tg)
    dts = mead - tg[0, :]                          # peak prominence over bin 1

    dz = abs(zd[1] - zd[0])
    imeadbv = np.round((zmead - zd[0]) / dz + 1.0 + btrk_below)
    lo, hi = min(btrk_range), max(btrk_range)
    with _quiet_nan():
        accept = ((dts > btrk_ts) & (zmead > lo) & (zmead < hi)
                  & (imeadbv < nbin - 1) & (imeadbv > 1))
    hbot = np.full(d.n_ens, np.nan)
    hbot[accept] = zmead[accept]
    return hbot
