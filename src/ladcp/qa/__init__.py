"""Acquisition quality-assessment layer (QA-first rebuild).

This subtree is intentionally independent of the velocity inverse. Each module computes
diagnostics from raw instrument fields (echo amplitude, correlation, attitude,
percent-good) and reports numbers validated against the MORIA-80 golden ``p``-struct.

See ``docs/PLAN_QA_REBUILD.md``.
"""
