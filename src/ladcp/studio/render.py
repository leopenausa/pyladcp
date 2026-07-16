"""Studio PNG rendering: QA panels and the Edit-view heatmap, with one shared cache."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import replace as _dc_replace

import numpy as np

from ..session import SessionConfig
from .payloads import _available_panels, _joint_n
from .state import StudioState

_MPL_LOCK = threading.Lock()          # matplotlib is not thread-safe
_PNG_CACHE: OrderedDict[tuple, bytes] = OrderedDict()
_PNG_CACHE_MAX = 48

HEAT_VIEWS = ("errvel", "echo")


def _cached(key: tuple) -> bytes | None:
    png = _PNG_CACHE.get(key)
    if png is not None:
        _PNG_CACHE.move_to_end(key)
    return png


def _render_png(key: tuple, make_fig, *, dpi=None, use_facecolor=False) -> bytes:
    """Cache-through PNG renderer: one lock, one Agg backend, one LRU for all views."""
    cached = _cached(key)
    if cached is not None:
        return cached
    import io as _io
    with _MPL_LOCK:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        fig = make_fig()
        kw: dict = {"format": "png"}
        if dpi is not None:
            kw["dpi"] = dpi
        if use_facecolor:
            kw["facecolor"] = fig.get_facecolor()
        buf = _io.BytesIO()
        fig.savefig(buf, **kw)
        plt.close(fig)
    png = buf.getvalue()
    _PNG_CACHE[key] = png
    while len(_PNG_CACHE) > _PNG_CACHE_MAX:
        _PNG_CACHE.popitem(last=False)
    return png


def render_panel(state: StudioState, label: str, panel: str,
                 cfg: SessionConfig) -> bytes:
    """One QA panel as PNG bytes for ``cfg`` (solve is ~30 ms warm; render dominates)."""
    key = (label, cfg, panel)
    png = _cached(key)
    if png is not None:
        return png

    with state.lock_for(label):
        ses = state.session(label)
        prep = ses.prepare(cfg.edit)
        result = ses.solve(cfg)

    from ..plots.alignment import alignment_figure
    from ..plots.btrack_figure import btrack_figure
    from ..plots.depth_figure import depth_figure
    from ..plots.drift_figure import drift_figure
    from ..plots.edit_figure import edit_figure
    from ..plots.error_figure import error_figure
    from ..plots.inverse_figure import constraint_weights_figure, inverse_diagnostics_figure
    from ..plots.raw_dashboard import raw_dashboard
    from ..plots.sadcp_figure import sadcp_figure
    from ..plots.shear_figure import shear_figure
    from ..plots.velocity_figure import velocity_figure

    if panel not in _available_panels(result):
        raise KeyError(f"panel {panel!r} not available for this configuration "
                       f"(have: {', '.join(_available_panels(result))})")
    makers = {
        "raw": lambda: raw_dashboard(prep.dh),
        "alignment": lambda: alignment_figure(prep.dh),
        "depth": lambda: depth_figure(prep.dh, prep.ctd),
        "edit": lambda: edit_figure(prep.dh, prep.ctd),
        "velocity": lambda: velocity_figure(result.vp, bottom=result.bp, station=label),
        "shear": lambda: shear_figure(result.shear, station=label),
        "inverse": lambda: inverse_diagnostics_figure(result, station=label),
        "weights": lambda: constraint_weights_figure(result.weights, station=label),
        "error": lambda: error_figure(result, station=label),
        "drift": lambda: drift_figure(result, station=label),
        "btrack": lambda: btrack_figure(result, station=label),
        "sadcp": lambda: sadcp_figure(result, station=label),
    }
    return _render_png(key, makers[panel], dpi=110, use_facecolor=True)


def render_heatmap(state: StudioState, label: str, head: str, view: str,
                   cfg: SessionConfig) -> bytes:
    """The per-head raw matrix (bins x joint-trimmed ensembles) as a PNG.

    Full-bleed (no axes/margins): the client maps pixels to (ensemble, bin)
    purely fractionally from ``edit/meta``. Auto-screened cells are dimmed so
    the user sees what the pipeline already rejects. Cached with the manual
    flags STRIPPED from the key -- a brush stroke never changes the base image
    (rectangles draw client-side), so brushing stays snappy.
    """
    key = (label, "_heatmap", head, view, _dc_replace(cfg.edit, manual_flags=()))
    png = _cached(key)
    if png is not None:
        return png

    with state.lock_for(label):
        ses = state.session(label)
        prep = ses.prepare(cfg.edit)
        dh = prep.dh
        h = dh.down if head == "down" else dh.up
        if h is None:
            raise KeyError(f"no {head}-looker in this configuration")
        n = _joint_n(dh)
        if view == "errvel":
            mat = np.abs(np.asarray(h.vel[3][:, :n], float))
        elif view == "echo":
            mat = np.nanmean(np.asarray(h.echo[:, :, :n], float), axis=0)
        else:
            raise KeyError(f"unknown heatmap view {view!r} (have: {', '.join(HEAT_VIEWS)})")
        from ..qa.screen import screen
        sr = screen(dh, prep.params)
        good = sr.good_down if head == "down" else sr.good_up
        auto = None if good is None else ~good[:, :n]

    from ..plots.edit_heatmap import edit_heatmap_figure
    return _render_png(key, lambda: edit_heatmap_figure(mat, auto=auto, view=view))
