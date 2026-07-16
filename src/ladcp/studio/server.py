"""The ``ladcp-studio`` server — split into focused modules; this is the stable façade.

* :mod:`.state` — :class:`StudioState` (station list, session LRU, journal access),
  the SADCP source labels/discovery.
* :mod:`.payloads` — request JSON ↔ :class:`~ladcp.session.SessionConfig` and the
  solve-response shaping.
* :mod:`.render` — QA-panel and Edit-view PNG rendering (one lock + LRU cache).
* :mod:`.app` — :func:`create_app`, the FastAPI routes.
* :mod:`.cli` — the ``ladcp-studio`` entry point (:func:`main`).

Endpoints (all JSON; profile arrays are NaN-sanitised to ``null`` for the browser):

* ``GET  /api/stations`` -- the launch work-list + whether a SADCP source is loaded.
* ``POST /api/station/{label}/prepare`` -- warm the edit-tier cache (~1.5 s cold).
* ``POST /api/station/{label}/solve`` -- solve a :class:`~ladcp.session.SessionConfig`
  on the cached context (~30 ms warm); the response carries the solved profile, the
  resolved declination, timings, the available QA panels, and the exact ``ladcp-qa``
  command line that reproduces it.
* ``POST /api/station/{label}/qa/{panel}`` -- one QA figure (the existing matplotlib
  panels: raw/depth/edit/velocity/weights/...) rendered as PNG for the posted config.
* ``POST /api/station/{label}/edit/meta`` / ``edit/heatmap/{head}/{view}`` -- the brush
  Edit view: grid geometry + the raw ensemble matrix (|errvel| or echo) as a full-bleed
  PNG the client maps fractionally.
* ``GET/POST/DELETE /api/station/{label}/edits[/{id}]`` -- the manual-edit journal
  (``<root>/.ladcp_edits/<station>.json``). The journal is the single source of truth:
  request bodies never carry rectangles; every config-consuming endpoint attaches the
  journal's flags server-side, so Studio solves, QA panels and the emitted
  ``ladcp-qa --edits`` command can never disagree.

The static single-page UI (no build step) is served from ``static/`` at ``/``.
"""
from __future__ import annotations

from .app import create_app  # noqa: F401
from .cli import main  # noqa: F401
from .payloads import (  # noqa: F401
    _arr,
    _num,
    config_from_body,
    solve_payload,
)
from .render import HEAT_VIEWS, render_heatmap, render_panel  # noqa: F401
from .state import (  # noqa: F401
    MAX_SESSIONS,
    StationEntry,
    StudioState,
    codas_label,
    discover_codas_products,
    merge_discovered_codas,
    raw_label,
)

__all__ = ["create_app", "main", "config_from_body", "solve_payload", "render_panel",
           "render_heatmap", "HEAT_VIEWS", "MAX_SESSIONS", "StationEntry", "StudioState",
           "codas_label", "discover_codas_products", "merge_discovered_codas", "raw_label"]

if __name__ == "__main__":
    raise SystemExit(main())
