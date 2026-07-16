"""The processing package: acquisition QA *and* the velocity solve.

Historically this subtree held only the quality-assessment layer; it has since grown
the full pipeline — ingest, screening/editing, CTD sync, seabed detection,
super-ensembles, and both velocity solvers (``inverse``/``inverse_full`` and the
shear method) — orchestrated by :mod:`.pipeline` and driven by the ``ladcp-qa``
command (:mod:`.cli`). The package name is kept for stability (imports, the
``ladcp-qa`` entry point, and ``source_stage`` tags in QA output all build on it).

Reading order and a module-by-module map: ``src/ladcp/README.md``.
"""
